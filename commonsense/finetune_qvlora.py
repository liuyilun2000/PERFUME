# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import copy
import json
import math
import os
import re
import sys
from os.path import join
from pathlib import Path
from typing import List, Optional, Union

import fire
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from tqdm import tqdm

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from mixtral_modification.modeling_mixtral import (
    MixtralAdapterForCausalLM,
    MixtralAdapterModel,
)

AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)


import peft
from peft import PeftModel, LoraConfig, get_peft_model

from utils import (
    get_adapter_args,
    init_trainable_parameters,
    convert_trainable_parameters,
    print_trainable_parameters,
)


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "test",
        output_dir: str = "./test",
        # load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        output_router_logits: bool = True,
        eval_step: int = 200,
        save_step: int = 200,
        # PERFUME hyperparams
        shared_adapter: bool = False,
        shared_adapter_num: int = 0,
        shared_routing_adapter: bool = False,
        shared_routing_adapter_num_experts: int = 0,
        shared_routing_adapter_num_experts_per_tok: int = 0,
        embedded_routing_adapter: bool = False,
        # PERFUME adapter hyperparams
        adapter_type: str = "",
        lora_r: int = 16,
        lora_alpha: int = 32,
        hidden_dim: int = 16,
        dropout: float = 0.05,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"output_router_logits: {output_router_logits}\n"
        f"adapter_type: {adapter_type}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"hidden_dim: {hidden_dim}\n"
        f"dropout: {dropout}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='mistralai/Mixtral-8x7B-Instruct-v0.1'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.output_router_logits = output_router_logits
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map='auto'#{"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    print(model.config)
    
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    print(model)
    
    adapter_args = get_adapter_args(adapter_type, lora_r, lora_alpha, hidden_dim, dropout)
    if adapter_type == "LoRA":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['gate', 'gate_proj', 'up_proj', 'down_proj'],#['gate'],#["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    model = get_peft_model(model, config)
    print(model)


    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    if resume_from_checkpoint:
        print(f"##### Loading checkpoint from {resume_from_checkpoint} #####")
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "model.safetensors"
        )  # Full checkpoint
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = load_file(checkpoint_name)

            def load_trainable_params(model, state_dict):
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].requires_grad}
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=False)
                return model

            model = load_trainable_params(model, adapters_weights)
            print(f"##### Successfully loaded trainable parameters from {checkpoint_name} #####")
            print_trainable_parameters(model)
        else:
            print(f"##### Checkpoint {checkpoint_name} not found #####")
    #else:
    #    print("##### Initializing parameters #####")
    #    init_trainable_parameters(model)
    
    #trainable_param_names = ['lora_A', 'lora_B', 'adapter_w1', 'adapter_w2', 'shared_routing_adapter_gate']
    #convert_trainable_parameters(model, trainable_param_names)

    print_trainable_parameters(model)
    #model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    model.enable_input_require_grads()
    model.train()
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    print(f"##### Trainer successfully initialized for {wandb_run_name} #####")
    model.config.use_cache = False

    old_state_dict = model.state_dict

    model.state_dict = (
        lambda self, *_, **__: {
            name: param.data
            for name, param in self.named_parameters()
            if param.requires_grad
        }
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)