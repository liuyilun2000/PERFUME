#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --job-name=Mixtral-8x7B.full.train
#SBATCH --output=log/Mixtral-8x7B.full.train.slurm.log
#SBATCH --mail-type=ALL




python finetune_full.py \
    --base_model 'mistralai/Mixtral-8x7B-v0.1' \
    --data_path 'commonsense_170k.json' \
    --output_dir 'checkpoints/Mixtral-8x7B-v0.1.full' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 1e-5  --cutoff_len 256 --val_set_size 120 \
    --eval_step 80  --save_step 80 \
    --wandb_project 'peft-moe' \
    --wandb_run_name 'Mixtral-8x7B.full' \
    | tee -a log/Mixtral-8x7B.full.train.log

