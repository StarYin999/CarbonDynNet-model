import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
from typing import Tuple, Dict, List, Optional
from model import SewerGNN
from data_utils import load_node_data
from train import split_data

class CarbonDynNetRLEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 gnn_model_path: str,
                 data_dir: str,
                 device: str = 'cpu',
                 max_episode_steps: int = 200,
                 controlled_node_ids: List[int] = None,
                 target_node_id: int = 7,
                 reward_weights: Dict[str, float] = None):
        super(CarbonDynNetRLEnv, self).__init__()
        self.device = torch.device(device)
        self.gnn_model_path = gnn_model_path
        self.data_dir = data_dir
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reward_weights = reward_weights or {"toc": 0.5, "doc": 2.0, "tn": 1.0}
        self.controlled_node_ids = controlled_node_ids if controlled_node_ids is not None else [13, 18, 28, 30, 33, 42, 61, 65]
        self.target_node_id = target_node_id
        self.gnn_model, self.node_mapping, self.input_dim, self.hidden_dim, self.output_dim, self.num_nodes = self._load_gnn_model()
        self.target_node_idx = self.node_mapping.get(self.target_node_id)
        self.controlled_node_indices = [self.node_mapping[nid] for nid in self.controlled_node_ids]
        feature_names = [f"feature_{i}" for i in range(self.input_dim)]
        self.toc_indices = [i for i, name in enumerate(feature_names) if name in ["feature_16", "feature_0", "feature_1", "feature_2", "feature_17"]]
        self.doc_indices = [i for i, name in enumerate(feature_names) if name in ["feature_0", "feature_1", "feature_2"]]
        self.tn_indices = [i for i, name in enumerate(feature_names) if name in ["feature_8", "feature_10", "feature_9"]]
        self.flow_rate_feature_idx = 29 if self.input_dim > 29 else 0
        self.normalized_flow_min = 0.0
        self.normalized_flow_max = 1.0
        self.action_space = spaces.Box(low=self.normalized_flow_min, high=self.normalized_flow_max, shape=(len(self.controlled_node_indices),), dtype=np.float32)
        state_dim = 4 + len(self.controlled_node_indices)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.node_features, _, _, _ = load_node_data(self.data_dir)
        self.historical_steps = self.node_features.shape[1]
        self.time_steps_gnn = 20 if self.historical_steps < 20 else 20
        self.current_gnn_input = None
        self.last_action = None

    def _load_gnn_model(self):
        checkpoint = torch.load(self.gnn_model_path, map_location=self.device)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        output_dim_gnn = checkpoint['output_dim']
        dropout_rate = checkpoint.get('dropout_rate', 0.5)
        node_mapping = checkpoint['node_mapping']
        num_nodes_gnn = len(node_mapping)
        model = SewerGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim_gnn, dropout_rate=dropout_rate, num_nodes=num_nodes_gnn)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, node_mapping, input_dim, hidden_dim, output_dim_gnn, num_nodes_gnn

    def get_state_metrics(self, node_features):
        target_features = node_features[self.target_node_idx, -1, :]
        toc = float(sum([target_features[i].item() for i in self.toc_indices]))
        doc = float(sum([target_features[i].item() for i in self.doc_indices]))
        tn = float(sum([target_features[i].item() for i in self.tn_indices]))
        doc_toc = doc / (toc + 1e-9)
        cn_ratio = toc / (tn + 1e-9)
        return toc, doc_toc, cn_ratio, tn

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.historical_steps <= self.time_steps_gnn:
            start_idx = 0
            gnn_input = self.node_features[:, start_idx:self.historical_steps, :].clone()
            if self.time_steps_gnn - gnn_input.shape[1] > 0:
                pad = torch.zeros((self.num_nodes, self.time_steps_gnn - gnn_input.shape[1], self.input_dim))
                gnn_input = torch.cat((gnn_input, pad), dim=1)
        else:
            start_idx = np.random.randint(0, self.historical_steps - self.time_steps_gnn)
            gnn_input = self.node_features[:, start_idx:start_idx+self.time_steps_gnn, :].clone()
        self.current_gnn_input = gnn_input
        with torch.no_grad():
            gnn_preds, _ = self.gnn_model(gnn_input, None)
        toc, doc_toc, cn_ratio, tn = self.get_state_metrics(gnn_preds)
        flow_rates = [self.current_gnn_input[idx, -1, self.flow_rate_feature_idx].item() for idx in self.controlled_node_indices]
        obs = np.array([toc, doc_toc, cn_ratio, tn] + flow_rates, dtype=np.float32)
        self.last_action = np.array(flow_rates, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.normalized_flow_min, self.normalized_flow_max)
        gnn_input = self.current_gnn_input.clone()
        for i, node_idx in enumerate(self.controlled_node_indices):
            gnn_input[node_idx, -1, self.flow_rate_feature_idx] = float(action[i])
        with torch.no_grad():
            gnn_preds, _ = self.gnn_model(gnn_input, None)
        toc, doc_toc, cn_ratio, tn = self.get_state_metrics(gnn_preds)
        flow_rates = [gnn_input[idx, -1, self.flow_rate_feature_idx].item() for idx in self.controlled_node_indices]
        obs = np.array([toc, doc_toc, cn_ratio, tn] + flow_rates, dtype=np.float32)
        reward = self.reward_weights["toc"] * toc + self.reward_weights["doc"] * doc_toc - self.reward_weights["tn"] * tn
        done = (self.current_step >= self.max_episode_steps)
        truncated = done
        info = {"toc": toc, "doc_toc": doc_toc, "cn_ratio": cn_ratio, "tn": tn}
        self.last_action = action
        self.current_gnn_input = gnn_input
        return obs, reward, False, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")

    def close(self):
        print("CarbonDynNet RL Environment closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--max_ep_steps', type=int, default=200)
    args = parser.parse_args()
    env = CarbonDynNetRLEnv(gnn_model_path=args.gnn_model_path, data_dir=args.data_dir, device=args.device, max_episode_steps=args.max_ep_steps)
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        print("Checking environment...")
        check_env(env)
        print("Starting PPO training...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000)
        print("Training finished.")
        obs, _ = env.reset()
        for _ in range(10):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Reward: {reward}, Info: {info}")
            if done or truncated:
                break
    except ImportError:
        print("stable_baselines3 not installed. Please install it to train PPO agent.")
