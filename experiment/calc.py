import os
import pandas as pd
def analyze_model_performance(directory, dataset, model=''):
    # Create a dictionary to store the model performance data
    performance_data = {}
    # Print header
    print("-" * 90)
    print(f"{'Filename':<60} {'Total':<10} {'Flagged':<10} {'Ratio':<10}")
    print("-" * 90)
    # Get and sort the list of JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    # Iterate through the sorted files in the directory
    for filename in json_files:
        if filename.endswith('.json') and dataset in filename and model in filename:
            with open(os.path.join(directory, filename), 'r') as f:
                data = pd.read_json(f)
            total = len(data['flag'])
            flagged = sum(data['flag'])
            ratio = flagged / total if total > 0 else 0
            print(f"{filename:<60} {total:<10} {flagged:<10} {ratio:.4f}")


# Call the function with the directory containing the JSON files
for dataset in ['boolq', 'piqa', 'social_i_qa', 'hellaswag', 'winogrande', 'ARC-Easy', 'ARC-Challenge', 'openbookqa']:
    analyze_model_performance('experiment', dataset, model='OLMoE')