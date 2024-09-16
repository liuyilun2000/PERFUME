#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=19:59:00
#SBATCH --job-name=OLMoE-1B-7B.lora_add.train
#SBATCH --output=log/OLMoE-1B-7B.lora_add.%j.train.slurm.log
#SBATCH --mail-type=ALL


declare -a configs=(
    "64 128 False True False 2 1 64.2-1"
    "64 128 False True False 2 2 64.2-2"
    "4 8 True False False 1 0 4.1"
    "2 4 False False True 0 0 2.e"
)



# Function to run training
run_training() {
    local lora_r=$1
    local lora_alpha=$2
    local shared_adapter=$3
    local shared_routing_adapter=$4
    local embedded_routing_adapter=$5
    local adapter_num_experts=$6
    local num_experts_per_tok=$7
    local run_name=$8
    local gpu_id=$9
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    
    local full_run_name="OLMoE-1B-7B.lora_${run_name}"
    local output_dir="checkpoints/OLMoE-1B-7B-0924.lora_${run_name}"
    local log_file="log/${full_run_name}.train.log"
    
    echo "Starting training for ${full_run_name} on GPU ${gpu_id}"
    
    python finetune.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'commonsense_170k.json' \
        --output_dir "${output_dir}" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 1e-5 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --shared_adapter ${shared_adapter} --shared_adapter_num ${adapter_num_experts} \
        --shared_routing_adapter ${shared_routing_adapter} --shared_routing_adapter_num_experts ${adapter_num_experts} \
        --shared_routing_adapter_num_experts_per_tok ${num_experts_per_tok} \
        --adapter_type 'LoRA' --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
        --embedded_routing_adapter ${embedded_routing_adapter} \
        --wandb_project 'peft-moe' \
        --wandb_run_name "${full_run_name}" \
        2>&1 | tee -a "${log_file}"
    
    echo "Finished training for ${full_run_name} on GPU ${gpu_id}"
}


# Run training tasks in parallel
for i in "${!configs[@]}"; do
    IFS=' ' read -ra config <<< "${configs[$i]}"
    run_training "${config[@]}" $i &
done

# Wait for all background jobs to finish
wait

echo "All training jobs completed."