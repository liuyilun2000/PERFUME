#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:59:00
#SBATCH --job-name=Mixtral-8x7B.eval
#SBATCH --mail-type=ALL

# Check if a model name argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a model name argument"
    exit 1
fi

MODEL_NAME=$1

#SBATCH --output=log/Mixtral-8x7B.${MODEL_NAME}.eval.slurm.log



run_evaluation() {
    local dataset=$1
    python commonsense_evaluate.py \
        --dataset $dataset \
        --base_model mistralai/Mixtral-8x7B-v0.1 \
        --peft_model checkpoints/Mixtral-8x7B-v0.1.${MODEL_NAME} \
        --name Mixtral-8x7B.${MODEL_NAME} \
        --batch_size 20 --max_new_tokens 4 \
        | tee -a log/Mixtral-8x7B.${MODEL_NAME}.eval.${dataset}.log
}

# Run evaluations for each dataset
#run_evaluation boolq
#run_evaluation piqa
#run_evaluation social_i_qa
run_evaluation hellaswag
#run_evaluation winogrande
#run_evaluation ARC-Challenge
#run_evaluation ARC-Easy
#run_evaluation openbookqa