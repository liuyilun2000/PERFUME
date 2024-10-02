#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=9:59:00
#SBATCH --job-name=Mixtral-8x7B.lora_16.1.train
#SBATCH --output=log/Mixtral-8x7B.lora_16.1.train.slurm.log
#SBATCH --mail-type=ALL




python finetune.py \
    --base_model 'mistralai/Mixtral-8x7B-v0.1' \
    --data_path 'commonsense_170k.json' \
    --output_dir 'checkpoints/Mixtral-8x7B-v0.1.lora_16.1' \
    --resume_from_checkpoint 'checkpoints/Mixtral-8x7B-v0.1.lora_16.1/checkpoint-800' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-5  --cutoff_len 256 --val_set_size 120 \
    --eval_step 80  --save_step 80 \
    --shared_adapter True  --shared_adapter_num 1 \
    --adapter_type 'LoRA'  --lora_r 16 --lora_alpha 32 \
    --wandb_project 'peft-moe' \
    --wandb_run_name 'Mixtral-8x7B.lora_16.1' \
    | tee -a log/Mixtral-8x7B.lora_16.1.train.log



    
    #