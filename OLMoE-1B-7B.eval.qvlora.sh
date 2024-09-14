#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:59:00
#SBATCH --job-name=OLMoE-1B-7B.qvlora.eval
#SBATCH --output=log/OLMoE-1B-7B.qvlora_%A_%a.eval.slurm.log
#SBATCH --mail-type=ALL

#SBATCH --array=0-3


declare -a configs=(
    "qvlora_4.aux"
    "qvlora_8.aux"
    "qvlora_16.aux"
    "qvlora_32.aux"
)



# Get the model configuration for this job array task
MODEL_NAME="${configs[$SLURM_ARRAY_TASK_ID]}"

# Set the GPU device for this task
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID




run_evaluation() {
    local dataset=$1
    python commonsense_evaluate.qvlora.py \
        --dataset $dataset \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --peft_model checkpoints/OLMoE-1B-7B-0924.${MODEL_NAME} \
        --name OLMoE-1B-7B.${MODEL_NAME} \
        --batch_size 16 --max_new_tokens 4 \
        | tee -a log/OLMoE-1B-7B.${MODEL_NAME}.eval.${dataset}.log
}


# Run evaluations for each dataset
datasets=(
    "boolq"
    "piqa"
    "social_i_qa"
    "winogrande"
    "ARC-Challenge"
    "ARC-Easy"
    "openbookqa"
    "hellaswag"
)

for dataset in "${datasets[@]}"; do
    run_evaluation $dataset
done

