#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --mail-type=ALL

'''
declare -a configs=(
    "OLMoE-1B-7B-0924"
    "OLMoE-1B-7B-0924-Instruct"
)
'''

# Function to run evaluation
run_evaluation() {
    local BASE_MODEL=$1
    local gpu_id=$2
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for dataset in "boolq" "piqa" "social_i_qa" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa" "hellaswag"; do
        python commonsense_evaluate.base.py \
            --dataset $dataset \
            --base_model "checkpoints/${BASE_MODEL}" \
            --name ${BASE_MODEL} \
            --batch_size 16 --max_new_tokens 4 \
            | tee -a log/${BASE_MODEL}.eval.${dataset}.log
    done
}

'''
# Run evaluations in parallel
for i in {0..1}; do
    run_evaluation "${configs[$i]}" $i &
done
'''
run_evaluation "OLMoE-1B-7B-0924.full" 0
# Wait for all background jobs to finish
wait