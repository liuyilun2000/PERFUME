#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --job-name=OLMoE-1B-7B.qvlora.train
#SBATCH --output=log/OLMoE-1B-7B.qvlora_%j.aux.train.slurm.log
#SBATCH --mail-type=ALL

'''
declare -a configs=(
    "4 8 OLMoE-1B-7B.qvlora_4"
    "8 16 OLMoE-1B-7B.qvlora_8"
    "16 32 OLMoE-1B-7B.qvlora_16"
    "32 64 OLMoE-1B-7B.qvlora_32"
)
declare -a configs=(
    "2 4 OLMoE-1B-7B.qvlora_2"
    "64 128 OLMoE-1B-7B.qvlora_64"
    "128 256 OLMoE-1B-7B.qvlora_128"
    "1 2 OLMoE-1B-7B.qvlora_1"
    "256 512 OLMoE-1B-7B.qvlora_256"
    "512 1024 OLMoE-1B-7B.qvlora_512"
    "1024 2048 OLMoE-1B-7B.qvlora_1024"
    "4 8 OLMoE-1B-7B.gatelora_4"
    "8 16 OLMoE-1B-7B.gatelora_8"
    "16 32 OLMoE-1B-7B.gatelora_16"
    "32 64 OLMoE-1B-7B.gatelora_32"
    "4 8 OLMoE-1B-7B.expertlora_4"
    "8 16 OLMoE-1B-7B.expertlora_8"
    "16 32 OLMoE-1B-7B.expertlora_16"
    "32 64 OLMoE-1B-7B.expertlora_32"
    "4 8 OLMoE-1B-7B.gateexpertlora_4"
    "8 16 OLMoE-1B-7B.gateexpertlora_8"
    "16 32 OLMoE-1B-7B.gateexpertlora_16"
)
'''
declare -a configs=(
    "4 8 OLMoE-1B-7B.expertlora_4 24960"
    "8 16 OLMoE-1B-7B.expertlora_8 24160"
    "16 32 OLMoE-1B-7B.expertlora_16 22000"
    "32 64 OLMoE-1B-7B.expertlora_32 22080"
    "4 8 OLMoE-1B-7B.gateexpertlora_4 23760"
    "8 16 OLMoE-1B-7B.gateexpertlora_8 24000"
    "16 32 OLMoE-1B-7B.gateexpertlora_16 24400"
)

# Function to run training
run_training() {
    local lora_r=$1
    local lora_alpha=$2
    local run_name=$3
    local ckp=$4
    #local gpu_id=$5
    
    # Set environment variable for this GPU
    # export CUDA_VISIBLE_DEVICES=$gpu_id
    
    python finetune_qvlora.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'commonsense_170k.json' \
        --output_dir "checkpoints/OLMoE-1B-7B-0924.gateexpertlora_${lora_r}.aux" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 2e-5 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --output_router_logits True \
        --adapter_type 'LoRA' --lora_r $lora_r --lora_alpha $lora_alpha \
        --wandb_project 'peft-moe' \
        --wandb_run_name "${run_name}.aux" \
        --resume_from_checkpoint "checkpoints/OLMoE-1B-7B-0924.gateexpertlora_${lora_r}.aux/checkpoint-${ckp}" \
        | tee -a "log/${run_name}.aux.train.log"
}


# Run training tasks in parallel
#for i in {0..3}; do
for i in {6..6}; do
    IFS=' ' read -ra config <<< "${configs[$i]}"
    run_training "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" #$i &
done

# Wait for all background jobs to finish
wait