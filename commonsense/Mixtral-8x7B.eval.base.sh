#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:59:00
#SBATCH --job-name=Mixtral-8x7B.eval
#SBATCH --mail-type=ALL
#SBATCH --output=log/Mixtral-8x7B.eval.slurm.log
#SBATCH --output=log/Mixtral-8x7B-Instruct.eval.slurm.log


BASE_MODEL=Mixtral-8x7B
#BASE_MODEL=Mixtral-8x7B-Instruct

run_evaluation() {
    local dataset=$1
    python commonsense_evaluate.base.py \
        --dataset $dataset \
        --base_model mistralai/${BASE_MODEL}-v0.1 \
        --name ${BASE_MODEL} \
        --batch_size 20 --max_new_tokens 4 \
        | tee -a log/${BASE_MODEL}.eval.${dataset}.log
}


# Run evaluations for each dataset
run_evaluation boolq
run_evaluation piqa
run_evaluation social_i_qa
run_evaluation hellaswag
run_evaluation winogrande
run_evaluation ARC-Challenge
run_evaluation ARC-Easy
run_evaluation openbookqa
