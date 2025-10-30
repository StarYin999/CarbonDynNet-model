import torch
import pandas as pd
import os
import numpy as np
from typing import List, Tuple, Dict

def load_node_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[int, int]]:
    node_mapping = {}
    current_idx = 0
    all_nodes = set()
    pipe_data = {}
    print("--- Pass 1: Reading pipe data and identifying nodes ---")
    num_features = -1
    processed_files_count = 0
    max_time_steps = 0
    pipe_features_dict = {}
    for file in sorted(os.listdir(data_dir)):
        if file.endswith('.xlsx') and not file.startswith('~$'):
            try:
                parts = file.replace('.xlsx', '').split('-')
                if len(parts) != 2 or not parts[0].startswith('pipe_'):
                    continue
                node1 = int(parts[0].split('_')[1])
                node2 = int(parts[1])
                all_nodes.add(node1)
                all_nodes.add(node2)
                df = pd.read_excel(os.path.join(data_dir, file))
                features = df.select_dtypes(include=[np.number]).values
                n_time = features.shape[0]
                pipe_features_dict[file] = features
                if n_time > max_time_steps:
                    max_time_steps = n_time
                if num_features == -1:
                    num_features = features.shape[1]
                elif features.shape[1] != num_features:
                    continue
                pipe_data[file] = {'u': node1, 'v': node2, 'features': features}
                processed_files_count += 1
            except Exception:
                continue
    if processed_files_count == 0:
        raise ValueError("No valid Excel files processed.")
    for node in sorted(list(all_nodes)):
        node_mapping[node] = current_idx
        current_idx += 1
    num_unique_nodes = len(node_mapping)
    node_features_time = torch.zeros((num_unique_nodes, max_time_steps, num_features), dtype=torch.float32)
    node_counts = torch.zeros((num_unique_nodes, max_time_steps), dtype=torch.int32)
    for file, data in pipe_data.items():
        u, v = data['u'], data['v']
        features_np = data['features']
        n_time = features_np.shape[0]
        if v in node_mapping:
            v_idx = node_mapping[v]
            node_features_time[v_idx, :n_time, :] += torch.FloatTensor(features_np)
            node_counts[v_idx, :n_time] += 1
    for i in range(num_unique_nodes):
        for t in range(max_time_steps):
            if node_counts[i, t] > 0:
                node_features_time[i, t, :] /= node_counts[i, t].float()
    mean_features = node_features_time.mean(dim=1, keepdim=True)
    std_features = node_features_time.std(dim=1, keepdim=True)
    max_features = node_features_time.max(dim=1, keepdim=True)[0]
    min_features = node_features_time.min(dim=1, keepdim=True)[0]
    diff_features = torch.diff(node_features_time, dim=1)
    mean_diff = diff_features.mean(dim=1, keepdim=True)
    std_diff = diff_features.std(dim=1, keepdim=True)
    node_features_time = torch.cat([
        node_features_time,
        mean_features,
        std_features,
        max_features,
        min_features,
        mean_diff,
        std_diff
    ], dim=1)
    mean = node_features_time.mean(dim=(0, 1), keepdim=True)
    std = node_features_time.std(dim=(0, 1), keepdim=True)
    node_features_time = (node_features_time - mean) / (std + 1e-8)
    return node_features_time, node_features_time, list(pipe_data.keys()), node_mapping
