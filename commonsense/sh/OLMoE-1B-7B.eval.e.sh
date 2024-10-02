#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=3:59:00
#SBATCH --job-name=OLMoE-1B-7B.lora.eval
#SBATCH --output=log/OLMoE-1B-7B.lora.%j.eval.slurm.log
#SBATCH --mail-type=ALL



declare -a configs=(
    "lora_4.e"
    "lora_8.e"
    "lora_16.e"
    "lora_32.e"
)

# Function to run evaluation
run_evaluation() {
    local config=$1
    local gpu_id=$2
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for dataset in "boolq" "piqa" "social_i_qa" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa" "hellaswag"; do
        python commonsense_evaluate.py \
            --dataset $dataset \
            --base_model allenai/OLMoE-1B-7B-0924 \
            --peft_model checkpoints/OLMoE-1B-7B-0924.${config} \
            --name OLMoE-1B-7B.${config} \
            --batch_size 16 --max_new_tokens 4 \
            | tee -a log/OLMoE-1B-7B.${config}.eval.${dataset}.log
    done
}

# Run evaluations in parallel
for i in {0..3}; do
    run_evaluation "${configs[$i]}" $i &
done

# Wait for all background jobs to finish
wait