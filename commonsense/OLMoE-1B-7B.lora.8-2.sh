#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=19:59:00
#SBATCH --job-name=OLMoE-1B-7B.lora.8-2.train
#SBATCH --output=log/OLMoE-1B-7B.lora.8-2.%j.train.slurm.log
#SBATCH --mail-type=ALL


declare -a configs=(
    "4 8 OLMoE-1B-7B.lora_4.8-2"
    "8 16 OLMoE-1B-7B.lora_8.8-2"
    "16 32 OLMoE-1B-7B.lora_16.8-2"
    "32 64 OLMoE-1B-7B.lora_32.8-2"
)


# Function to run training
run_training() {
    local lora_r=$1
    local lora_alpha=$2
    local run_name=$3
    local gpu_id=$4
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    python finetune.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'commonsense_170k.json' \
        --output_dir "checkpoints/OLMoE-1B-7B-0924.lora_${lora_r}.8-2" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 1e-5 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --shared_routing_adapter True  --shared_routing_adapter_num_experts 8 \
        --shared_routing_adapter_num_experts_per_tok 2 \
        --adapter_type 'LoRA' --lora_r $lora_r --lora_alpha $lora_alpha \
        --wandb_project 'peft-moe' \
        --wandb_run_name "${run_name}" \
        | tee -a "log/${run_name}.train.log"
}

# Run training tasks in parallel
for i in {0..3}; do
    IFS=' ' read -ra config <<< "${configs[$i]}"
    run_training "${config[0]}" "${config[1]}" "${config[2]}" $i &
done

# Wait for all background jobs to finish
wait