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
config = "lora_16.2-2"
peft_model = f"checkpoints/OLMoE-1B-7B-0924.{config}"

tokenizer, model = load_model(base_model, peft_model)






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

# Usage example:
layer_index = 0  # Change this to extract vectors from a different layer
layer = model.model.layers[layer_index]
extracted_vectors = extract_layer_vectors(layer, layer_index)

# Now you can access the extracted vectors like this:
print("Router vectors shape:", extracted_vectors['router_vectors'].shape)
print("Number of FFN expert vectors:", len(extracted_vectors['ffn_expert_vectors']), extracted_vectors['ffn_expert_vectors'][0].shape)
print("Shared routing gate vectors shape:", extracted_vectors['shared_routing_gate_vectors'].shape)
print("Number of shared routing adapter vectors:", len(extracted_vectors['shared_routing_adapter_vectors']), extracted_vectors['shared_routing_adapter_vectors'][0].shape)



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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import umap_ as UMAP
import torch
from tqdm import tqdm

def prepare_data(extracted_vectors, sample_ratio=1.0, random_state=42):
    router_vectors = extracted_vectors['router_vectors'].to(torch.float32).detach().cpu().numpy()
    ffn_expert_vectors = [expert.to(torch.float32).detach().cpu().numpy() for expert in extracted_vectors['ffn_expert_vectors']]
    shared_routing_gate_vectors = extracted_vectors['shared_routing_gate_vectors'].to(torch.float32).detach().cpu().numpy()
    shared_routing_adapter_vectors = [adapter.to(torch.float32).detach().cpu().numpy() for adapter in extracted_vectors['shared_routing_adapter_vectors']]
    #
    data = []
    rng = np.random.RandomState(random_state)
    # FFN expert vectors (for learning transformations)
    ffn_data = []
    for i, expert_vectors in enumerate(ffn_expert_vectors):
        num_vectors = len(expert_vectors)
        num_samples = int(num_vectors * sample_ratio)
        if num_samples < num_vectors:
            sampled_indices = rng.choice(num_vectors, num_samples, replace=False)
            sampled_vectors = expert_vectors[sampled_indices]
        else:
            sampled_vectors = expert_vectors
        ffn_data.extend(sampled_vectors)
        for vec in sampled_vectors:
            data.append({'vector': vec, 'expert': i, 'type': 'FFN Expert'})
    # Router vectors
    for i, vec in enumerate(router_vectors):
        data.append({'vector': vec, 'expert': i, 'type': 'Router'})
    # Shared routing gate vectors
    for i, vec in enumerate(shared_routing_gate_vectors):
        data.append({'vector': vec, 'expert': i, 'type': 'Shared Routing Gate'})
    # Shared routing adapter vectors
    for i, adapter_vectors in enumerate(shared_routing_adapter_vectors):
        for vec in adapter_vectors:
            data.append({'vector': vec, 'expert': i, 'type': 'Shared Routing Adapter'})
    df = pd.DataFrame(data)
    all_vectors = np.stack(df['vector'].values)
    ffn_vectors = np.stack(ffn_data)
    scaler = StandardScaler()
    ffn_vectors_normalized = scaler.fit_transform(ffn_vectors)
    all_vectors_normalized = scaler.transform(all_vectors)
    print(f"Total vectors: {len(all_vectors)}")
    print(f"FFN vectors for learning: {len(ffn_vectors)}")
    return df, all_vectors_normalized, ffn_vectors_normalized

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

def visualize_tsne(df, proj_2d, selected_indices, n_pca_components, perplexity, learning_rate, n_iter):
    num_experts = len(selected_indices)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_experts))
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(selected_indices):
        expert_points = df[(df['expert'] == idx) & (~df['is_router'])]
        router_point = df[(df['expert'] == idx) & (df['is_router'])]
        print(f"Expert {idx}: {len(expert_points)} points, Router {idx}: {len(router_point)} point")
        plt.scatter(proj_2d[expert_points.index, 0], proj_2d[expert_points.index, 1], 
                    c=[colors[i]], marker='o', s=20, alpha=0.5, label=f'Expert {idx}')
        plt.scatter(proj_2d[router_point.index, 0], proj_2d[router_point.index, 1], 
                    c=[colors[i]], marker='*', s=200, edgecolors='black', label=f'Router {idx}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'PCA + t-SNE projection\n(PCA components: {n_pca_components}, perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter})')
    plt.tight_layout()
    plt.savefig(f'pca_tsne_visualization_c{n_pca_components}_p{perplexity}_lr{learning_rate}_i{n_iter}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_umap(df, proj_2d, n_pca_components, n_neighbors, min_dist, n_components):
    type_markers = {'FFN Expert': 'o', 'Router': '*', 'Shared Routing Gate': 's', 'Shared Routing Adapter': '^'}
    type_colors = {'FFN Expert': 'blue', 'Router': 'red', 'Shared Routing Gate': 'green', 'Shared Routing Adapter': 'purple'}
    plt.figure(figsize=(16, 12))
    for vec_type in type_markers.keys():
        mask = df['type'] == vec_type
        plt.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                    marker=type_markers[vec_type], c=type_colors[vec_type], 
                    s=50 if vec_type == 'FFN Expert' else 100, 
                    alpha=0.7 if vec_type == 'FFN Expert' else 1,
                    label=vec_type)
    plt.title(f'PCA + UMAP projection\n(PCA components: {n_pca_components}, n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components})')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6)
    plt.tight_layout()
    plt.savefig(f'pca_umap_visualization_c{n_pca_components}_n{n_neighbors}_d{min_dist}_c{n_components}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Usage
layer_index = 0
layer = model.model.layers[layer_index]
extracted_vectors = extract_layer_vectors(layer, layer_index)

# Prepare data with sampling
df, all_vectors_normalized, ffn_vectors_normalized = prepare_data(extracted_vectors, sample_ratio=0.5)

# Apply PCA
pca, n_pca_components = apply_pca(ffn_vectors_normalized, 0.8)
pca_result_all = pca.transform(all_vectors_normalized)

hyperparameters = [
    (10, 0.1, 2),   
    (10, 0.5, 2),   
    (10, 0.8, 2),   
]

for n_neighbors, min_dist, n_components in tqdm(hyperparameters, desc="Processing UMAP visualizations"):
    umap = UMAP.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    umap.fit(pca.transform(ffn_vectors_normalized))
    proj_2d = umap.transform(pca_result_all)
    visualize_umap(df, proj_2d, n_pca_components, n_neighbors, min_dist, n_components)
    print(f"Visualization saved for n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")

print("All visualizations completed.")