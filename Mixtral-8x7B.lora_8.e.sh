#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:59:00
#SBATCH --job-name=Mixtral-8x7B.lora_8.e.train
#SBATCH --output=log/Mixtral-8x7B.lora_8.e.train.slurm.log
#SBATCH --mail-type=ALL




python finetune.py \
    --base_model 'mistralai/Mixtral-8x7B-v0.1' \
    --data_path 'commonsense_170k.json' \
    --output_dir 'checkpoints/Mixtral-8x7B-v0.1.lora_8.e' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-5  --cutoff_len 256 --val_set_size 120 \
    --eval_step 80  --save_step 80 \
    --embedded_routing_adapter True \
    --adapter_type 'LoRA'  --lora_r 8 --lora_alpha 16 \
    --wandb_project 'peft-moe' \
    --wandb_run_name 'Mixtral-8x7B.lora_8.e' \
    | tee -a log/Mixtral-8x7B.lora_8.e.train.log



    
    #