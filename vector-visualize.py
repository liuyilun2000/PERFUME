import argparse
import torch
import transformers
from transformers import (
    GenerationConfig,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from mixtral_modification.modeling_mixtral import (
    MixtralAdapterForCausalLM,
    MixtralAdapterModel,
)

AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)


from olmoe_modification.configuration_olmoe import OlmoeAdapterConfig
from olmoe_modification.modeling_olmoe import (
    OlmoeAdapterForCausalLM,
    OlmoeAdapterModel,
)

AutoConfig.register("olmoe-adapter", OlmoeAdapterConfig)
AutoModel.register(OlmoeAdapterConfig, OlmoeAdapterModel)
AutoModelForCausalLM.register(OlmoeAdapterConfig, OlmoeAdapterForCausalLM)



from utils import (
    get_adapter_args,
    init_trainable_parameters,
    convert_trainable_parameters,
    print_trainable_parameters,
)

from safetensors import safe_open
from safetensors.torch import load_file

def load_model(base_model, peft_model):
    config_path = f"{peft_model}/config.json"
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config.output_router_logits = False
    #
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    #
    model = OlmoeAdapterForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    #
    checkpoint_name = f"{peft_model}/model.safetensors"
    if torch.cuda.is_available():
        adapters_weights = load_file(checkpoint_name, device="cuda")
    else:
        adapters_weights = load_file(checkpoint_name)
    #
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in adapters_weights.items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"##### Successfully loaded parameters from {checkpoint_name} #####")
    model.eval()
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



base_model = "allenai/OLMoE-1B-7B-0924"
#config = ["lora_8.1", "lora_8.4", "lora_8.4-1", "lora_8.4-4"]
config = "lora_16.8-2"
math_peft_model = f"math/checkpoints/OLMoE-1B-7B-0924.{config}"
commonsense_peft_model = f"commonsense/checkpoints/OLMoE-1B-7B-0924.{config}"

tokenizer, math_model = load_model(base_model, math_peft_model)

tokenizer, commonsense_model = load_model(base_model, commonsense_peft_model)






def extract_layer_vectors(layer, layer_index):
    # Extract router's expert vectors
    router_vectors = layer.mlp.gate.weight
    # Extract FFN expert memory vectors
    ffn_expert_vectors = [expert.up_proj.weight for expert in layer.mlp.experts]
    # Extract shared routing adapter gate vectors
    shared_routing_gate_vectors = layer.mlp.shared_routing_adapter_gate.weight
    # Extract shared routing adapter expert vectors
    shared_routing_adapter_vectors = [
        adapter.unit.lora_A.weight 
        for adapter in layer.mlp.shared_routing_adapter
    ]
    return {
        'router_vectors': router_vectors,
        'ffn_expert_vectors': ffn_expert_vectors,
        'shared_routing_gate_vectors': shared_routing_gate_vectors,
        'shared_routing_adapter_vectors': shared_routing_adapter_vectors
    }

'''
# Usage example:
layer_index = 0  # Change this to extract vectors from a different layer
layer = model.model.layers[layer_index]
extracted_vectors = extract_layer_vectors(layer, layer_index)

# Now you can access the extracted vectors like this:
print("Router vectors shape:", extracted_vectors['router_vectors'].shape)
print("Number of FFN expert vectors:", len(extracted_vectors['ffn_expert_vectors']), extracted_vectors['ffn_expert_vectors'][0].shape)
print("Shared routing gate vectors shape:", extracted_vectors['shared_routing_gate_vectors'].shape)
print("Number of shared routing adapter vectors:", len(extracted_vectors['shared_routing_adapter_vectors']), extracted_vectors['shared_routing_adapter_vectors'][0].shape)
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from umap import umap_ as UMAP

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import umap_ as UMAP
import torch
from tqdm import tqdm


def prepare_data_for_comparison(commonsense_vectors, math_vectors, sample_ratio=1.0, random_state=42):
    all_data = []
    # Process original MoE data (FFN Expert and Router)
    for i, expert_vectors in enumerate(commonsense_vectors['ffn_expert_vectors']):
        for vec in expert_vectors:
            all_data.append({'vector': vec.detach().to(torch.float32).cpu().numpy(), 'expert': i, 'type': 'FFN Expert'})
    for i, vec in enumerate(commonsense_vectors['router_vectors']):
        all_data.append({'vector': vec.detach().to(torch.float32).cpu().numpy(), 'expert': i, 'type': 'FFN Router'})
    # Process adapter data
    for model_name, vectors in [("Commonsense", commonsense_vectors), ("Math", math_vectors)]:
        for i, adapter_vectors in enumerate(vectors['shared_routing_adapter_vectors']):
            for vec in adapter_vectors:
                all_data.append({'vector': vec.detach().to(torch.float32).cpu().numpy(), 'expert': i, 'type': f'{model_name} Routing Adapter'})
        for i, vec in enumerate(vectors['shared_routing_gate_vectors']):
            all_data.append({'vector': vec.detach().to(torch.float32).cpu().numpy(), 'expert': i, 'type': f'{model_name} Routing Gate'})
    # Convert to DataFrame
    all_df = pd.DataFrame(all_data)
    # Normalize all vectors together
    all_vectors = np.stack(all_df['vector'].values)
    scaler = StandardScaler()
    all_vectors_normalized = scaler.fit_transform(all_vectors)
    # Split into original MoE and adapter data
    original_moe_mask = all_df['type'].isin(['FFN Expert'])
    original_moe_df = all_df[original_moe_mask].copy()
    adapter_df = all_df[~original_moe_mask].copy()
    # Update the normalized vectors in the DataFrames
    all_df['vector_normalized'] = list(all_vectors_normalized)
    original_moe_df['vector_normalized'] = list(all_vectors_normalized[original_moe_mask])
    adapter_df['vector_normalized'] = list(all_vectors_normalized[~original_moe_mask])
    original_moe_vectors_normalized = all_vectors_normalized[original_moe_mask]
    print(f"Total vectors: {len(all_vectors)}, Original MoE vectors: {len(original_moe_vectors_normalized)}")
    return all_df, all_vectors_normalized, original_moe_df, original_moe_vectors_normalized, adapter_df


def apply_pca(data, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= n_components) + 1
    #
    plt.figure(figsize=(10, 5))
    plt.plot(cumsum)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.savefig('pca_cumulative_variance.png')
    plt.close()
    print(f"Number of components explaining {n_components*100}% of variance: {d}")
    return pca, d  # Return the PCA object and the number of components



def visualize_model_comparison(all_df, proj_2d, n_pca_components, n_neighbors, min_dist, save_name):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'QTOptimum'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    type_markers = {
        'FFN Expert': 'o', 'FFN Router': '*',
        'Commonsense Routing Gate': '*', 'Commonsense Routing Adapter': 'o',
        'Math Routing Gate': '*', 'Math Routing Adapter': 'o'
    }
    point_sizes = {
        'FFN Expert': 5, 'FFN Router': 100,
        'Commonsense Routing Gate': 100, 'Commonsense Routing Adapter': 10,
        'Math Routing Gate': 100, 'Math Routing Adapter': 10
    }
    point_alpha = {
        'FFN Expert': 0.1, 'FFN Router': 1,
        'Commonsense Routing Gate': 1, 'Commonsense Routing Adapter': 0.4,
        'Math Routing Gate': 1, 'Math Routing Adapter': 0.4
    }
    def plot_data(ax, data, title):
        num_experts = data['expert'].nunique()
        color_palette = sns.color_palette("rainbow", n_colors=num_experts)
        for vec_type in data['type'].unique():
            for expert in data['expert'].unique():
                mask = (data['type'] == vec_type) & (data['expert'] == expert)
                ax.scatter(proj_2d[data.index[mask], 0], proj_2d[data.index[mask], 1],
                           marker=type_markers[vec_type],
                           c=[color_palette[expert]],
                           s=point_sizes[vec_type],
                           alpha=point_alpha[vec_type],
                           label=f'{vec_type} (Expert {expert})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.8)
        ax.set_title(title, fontsize=24, fontweight='bold')
    # Determine the overall data range based on the first subplot (original_moe_df)
    x_min, x_max = proj_2d[all_df.index, 0].min(), proj_2d[all_df.index, 0].max()
    y_min, y_max = proj_2d[all_df.index, 1].min(), proj_2d[all_df.index, 1].max()
    # Add some padding to the limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    # Plot data and set limits for each subplot
    ffn_df = all_df[all_df['type'].str.startswith('FFN')]
    plot_data(ax1, ffn_df, 'FFN Experts and Router')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('UMAP Dimension 1', fontsize=22, fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontsize=22, fontweight='bold')
    commonsense_adapter_df = all_df[all_df['type'].str.startswith('Commonsense')]
    plot_data(ax2, commonsense_adapter_df, 'Commonsense Adapters')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('UMAP Dimension 1', fontsize=22, fontweight='bold')
    math_adapter_df = all_df[all_df['type'].str.startswith('Math')]
    plot_data(ax3, math_adapter_df, 'Math Adapters')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel('UMAP Dimension 1', fontsize=22, fontweight='bold')
    fig.suptitle(f'PCA + UMAP Projection\n(PCA components: {n_pca_components}, n_neighbors={n_neighbors}, min_dist={min_dist})',
                 fontsize=28, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_name}_pca_c{n_pca_components}_umap_n{n_neighbors}_d{min_dist}.png',
                format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
# Usage

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'QTOptimum'



n_neighbors, min_dist, n_components = (20, 0.5, 2)

for layer_index in [8]:
    commonsense_layer = commonsense_model.model.layers[layer_index]
    math_layer = math_model.model.layers[layer_index]
    commonsense_vectors = extract_layer_vectors(commonsense_layer, layer_index)
    math_vectors = extract_layer_vectors(math_layer, layer_index)
    all_data, all_vectors_normalized, original_moe_df, original_moe_vectors_normalized, adapter_df = prepare_data_for_comparison(commonsense_vectors, math_vectors)
    pca, n_pca_components = apply_pca(original_moe_vectors_normalized, 0.8)
    pca_result_original = pca.transform(original_moe_vectors_normalized)
    pca_result_all = pca.transform(all_vectors_normalized)
    umap_model = UMAP.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    umap_model.fit(pca_result_original)
    proj_2d = umap_model.transform(pca_result_all)
    visualize_model_comparison(all_data, proj_2d, n_pca_components, n_neighbors, min_dist, f"layer_{layer_index}")


print("All visualizations completed.")