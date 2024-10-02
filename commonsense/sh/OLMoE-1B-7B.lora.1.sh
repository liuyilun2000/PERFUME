#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:59:00
#SBATCH --job-name=OLMoE-1B-7B.lora.1.train
#SBATCH --output=log/OLMoE-1B-7B.lora_%A_%a.1.train.slurm.log
#SBATCH --mail-type=ALL


#SBATCH --array=0-3


declare -a configs=(
    "8 16 OLMoE-1B-7B.lora_8.1"
    "16 32 OLMoE-1B-7B.lora_16.1"
    "32 64 OLMoE-1B-7B.lora_32.1"
    "64 128 OLMoE-1B-7B.lora_64.1"
)


# Read the configuration for this job array task
IFS=' ' read -r lora_r lora_alpha run_name <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

# Set the GPU device for this task
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID



python finetune.py \
    --base_model 'allenai/OLMoE-1B-7B-0924' \
    --data_path 'commonsense_170k.json' \
    --output_dir "checkpoints/OLMoE-1B-7B-0924.lora_${lora_r}.1" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 1e-5 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --shared_adapter True  --shared_adapter_num 1 \
    --adapter_type 'LoRA' --lora_r $lora_r --lora_alpha $lora_alpha \
    --wandb_project 'peft-moe' \
    --wandb_run_name "${run_name}" \
    | tee -a "log/${run_name}.train.log"




