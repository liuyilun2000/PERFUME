#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1:59:00
#SBATCH --job-name=OLMoE-1B-7B.qvlora.eval
#SBATCH --output=log/OLMoE-1B-7B.qvlora_%j.eval.slurm.log
#SBATCH --mail-type=ALL


#salloc -p accelerated -t 8:00:00 --gres=gpu:1

cd PERFUME
cd commonsense

declare -a configs=(
    "expertlora_4.aux"
    "expertlora_8.aux"
    "expertlora_16.aux"
    "expertlora_32.aux"
    "gateexpertlora_4.aux"
    "gateexpertlora_8.aux"
    "gateexpertlora_16.aux"
)
'''

declare -a configs=(
    "expertlora_4.aux"
    "expertlora_8.aux"
    "expertlora_16.aux"
    "expertlora_32.aux"
    "gatelora_4.aux"
    "gatelora_8.aux"
    "gatelora_16.aux"
    "gatelora_32.aux"
    "qvlora_1.aux"
    "qvlora_256.aux"
    "qvlora_512.aux"
    "qvlora_2.aux"
    "qvlora_64.aux"
    "qvlora_128.aux"
    "qvlora_64"
)

'''


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

run_evaluation() {
    local model_name=$1
    #local gpu_id=$2
    
    # Set environment variable for this GPU
    #export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for dataset in "${datasets[@]}"; do
        python commonsense_evaluate.qvlora.py \
            --dataset $dataset \
            --base_model allenai/OLMoE-1B-7B-0924 \
            --peft_model checkpoints/OLMoE-1B-7B-0924.${model_name} \
            --name OLMoE-1B-7B.${model_name} \
            --batch_size 16 --max_new_tokens 4 \
            | tee -a log/OLMoE-1B-7B.${model_name}.eval.${dataset}.log
    done
}

# Run evaluation tasks in parallel
for i in {1..1}; do
    run_evaluation "${configs[$i]}" $i &
done

# Wait for all background jobs to finish
wait