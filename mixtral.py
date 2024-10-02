# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


#TRANSFORMERS_CACHE_DIR = "/home/hk-project-p0022189/hgf_mxv5488/.cache/huggingface/transformers"
HF_TOKEN = "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"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
    #cache_dir=TRANSFORMERS_CACHE_DIR, 
    token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
    #cache_dir=TRANSFORMERS_CACHE_DIR, 
    token=HF_TOKEN)







def convert_trainable_parameters(model, trainable_param_names):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += 1
        if any(substring in name for substring in trainable_param_names):
            param.requires_grad = True
            trainable_params += 1
        else:
            param.requires_grad = False
    print(
        f"Convert trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Print trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f} || {all_param-trainable_params}"
    )



trainable_param_names = ['shared_routing_adapter_gate']
convert_trainable_parameters(model, trainable_param_names)

print_trainable_parameters(model)






import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

import requests
import json

import torch
from pathlib import Path
import os
from os.path import join
import copy
import argparse
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel

from mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from mixtral_modification.modeling_mixtral import MixtralAdapterModel, MixtralAdapterForCausalLM


AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)


shared_adapter=True 
shared_adapter_num=1
adapter_type='LoRA'
lora_r=16
lora_alpha=32
hidden_dim=None
dropout=0.05

def get_adapter_args(adapter_type, lora_r, lora_alpha, hidden_dim, dropout):
    adapter_configs = {
        'LoRA': {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': dropout
        },
        'Parallel_Adapter': {
            'hidden_dim': hidden_dim,
            'hidden_act': 'silu',
            'dropout': dropout
        }
    }    
    return adapter_configs.get(adapter_type, {})


adapter_args = get_adapter_args(adapter_type, lora_r, lora_alpha, hidden_dim, dropout)



config = MixtralAdapterConfig(
    vocab_size=32000,
    shared_adapter=True,
    shared_adapter_num=1,
    adapter_type=adapter_type,
    adapter_args=adapter_args,
    output_router_logits=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model, 
trust_remote_code=True)
print(tokenizer)
model = MixtralAdapterForCausalLM.from_pretrained(
    base_model,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map='cpu'#{"": int(os.environ.get("LOCAL_RANK") or 0)}
)