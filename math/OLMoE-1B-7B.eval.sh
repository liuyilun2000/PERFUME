#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=20:59:00
#SBATCH --mail-type=ALL
#SBATCH --job-name=OLMoE-1B-7B.lora.eval
#SBATCH --output=log/OLMoE-1B-7B.lora.%j.eval.slurm.log

'''
# Check if the model type parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Model type parameter is required."
    echo "Usage: sbatch $0 <model_type>"
    exit 1
fi

# Get the model type from the command line argument
MODEL_TYPE=$1

# Set job name and output file dynamically

# Define the configurations
'''
declare -a configs=(
    "lora_4.2-2"
    "lora_8.2-2"
    "lora_16.2-2"
    "lora_32.2-2"
)

# Function to run evaluation
run_evaluation() {
    local config=$1
    local gpu_id=$2
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for dataset in "MultiArith" "gsm8k" "AddSub" "AQuA" "SingleEQ" "SVAMP"; do
        python math_evaluate.py \
            --dataset $dataset \
            --base_model allenai/OLMoE-1B-7B-0924 \
            --peft_model checkpoints/OLMoE-1B-7B-0924.${config} \
            --name OLMoE-1B-7B.${config} \
            --batch_size 1 --max_new_tokens 256 \
            | tee -a log/OLMoE-1B-7B.${config}.eval.${dataset}.log
    done
}

# Run evaluations in parallel
for i in {0..3}; do
    run_evaluation "${configs[$i]}" $i &
done

# Wait for all background jobs to finish
wait

