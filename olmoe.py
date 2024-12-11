# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


#TRANSFORMERS_CACHE_DIR = "/home/hk-project-p0022189/hgf_mxv5488/.cache/huggingface/transformers"
HF_TOKEN = "*"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"
MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
    #cache_dir=TRANSFORMERS_CACHE_DIR, 
    token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
    #cache_dir=TRANSFORMERS_CACHE_DIR, 
    token=HF_TOKEN)




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


from olmoe_modification.configuration_olmoe import OlmoeAdapterConfig
from olmoe_modification.modeling_olmoe import (
    OlmoeAdapterForCausalLM,
    OlmoeAdapterModel,
)

AutoConfig.register("olmoe-adapter", OlmoeAdapterConfig)
AutoModel.register(OlmoeAdapterConfig, OlmoeAdapterModel)
AutoModelForCausalLM.register(OlmoeAdapterConfig, OlmoeAdapterForCausalLM)


from utils import (
    get_adapter_args,
    init_trainable_parameters,
    convert_trainable_parameters,
    print_trainable_parameters,
)



base_model = "allenai/OLMoE-1B-7B-0924"

config = OlmoeAdapterConfig(
    intermediate_size=1024,
    shared_adapter=True,
    shared_adapter_num=1,
    adapter_type='LoRA',
    adapter_args={
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05
    },
    output_router_logits=True
)
print(config)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = OlmoeAdapterForCausalLM.from_pretrained(
    base_model,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map='auto'#{"": int(os.environ.get("LOCAL_RANK") or 0)},
)






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
        f"Print trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



trainable_param_names = ['experts']
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
