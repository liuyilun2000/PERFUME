#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=152
#SBATCH --gres=gpu:4
#SBATCH --time=20:59:00
#SBATCH --job-name=OLMoE-1B-7B.train
#SBATCH --output=log/OLMoE-1B-7B.%j.train.slurm.log
#SBATCH --mail-type=ALL



run_1gpu_task() {
    export CUDA_VISIBLE_DEVICES=0
    python finetune_qvlora.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'commonsense_170k.json' \
        --output_dir "checkpoints/OLMoE-1B-7B-0924.qvlora_64" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-5 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --adapter_type 'LoRA' --lora_r 64 --lora_alpha 128 \
        --wandb_project 'peft-moe' \
        --wandb_run_name "OLMoE-1B-7B.qvlora_64" \
        | tee -a "log/OLMoE-1B-7B.qvlora_64.train.log" &
}


run_3gpu_task() {
    export CUDA_VISIBLE_DEVICES=1,2,3
    python finetune_full.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'commonsense_170k.json' \
        --output_dir "checkpoints/OLMoE-1B-7B-0924.full" \
        --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 1e-5  --cutoff_len 256 --val_set_size 120 \
        --eval_step 80  --save_step 80 \
        --wandb_project 'peft-moe' \
        --wandb_run_name 'OLMoE-1B-7B-0924.full' \
        | tee -a log/OLMoE-1B-7B-0924.full.train.log &
}


# Run both tasks in parallel
run_1gpu_task
run_3gpu_task

# Wait for all background jobs to complete
wait

echo "All tasks completed."