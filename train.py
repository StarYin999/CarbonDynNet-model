import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from model import WarmupCosineScheduler

def split_data(node_features: torch.Tensor, target_features: torch.Tensor, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
    num_nodes, time_steps, num_features = node_features.shape
    target_time_steps = target_features.shape[1]
    time_steps = min(time_steps, target_time_steps)
    node_features = node_features[:, :time_steps, :]
    target_features = target_features[:, :time_steps, :]
    train_steps = int(time_steps * train_ratio)
    val_steps = int(time_steps * val_ratio)
    test_steps = time_steps - train_steps - val_steps
    train_data = Data(x=node_features[:, :train_steps, :], edge_index=None)
    train_target = target_features[:, :train_steps, :]
    val_data = Data(x=node_features[:, train_steps:train_steps + val_steps, :], edge_index=None)
    val_target = target_features[:, train_steps:train_steps + val_steps, :]
    test_data = Data(x=node_features[:, train_steps + val_steps:, :], edge_index=None)
    test_target = target_features[:, train_steps + val_steps:, :]
    return train_data, val_data, test_data, train_target, val_target, test_target

def evaluate_model(model: nn.Module, data: Data, target: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        predictions, _ = model(data.x, data.edge_index)
    y_true = target.numpy()
    y_pred = predictions.numpy()
    metrics = {'overall': {}, 'features': {}}
    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else float('inf')
        return {'mae': mae, 'mse': mse, 'r2': r2, 'mape': mape}
    for i in range(y_true.shape[2]):
        feature_true = y_true[:, :, i].flatten()
        feature_pred = y_pred[:, :, i].flatten()
        metrics['features'][f'feature_{i}'] = calculate_metrics(feature_true, feature_pred)
    metrics['overall'] = calculate_metrics(y_true.flatten(), y_pred.flatten())
    return metrics

def save_metrics_to_excel(metrics: dict, save_path: str):
    overall_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'MAPE', 'R2'],
        'Value': [
            metrics['overall']['mae'],
            metrics['overall']['mse'],
            metrics['overall']['mape'],
            metrics['overall']['r2']
        ]
    })
    features_data = []
    for feature_name, feature_metrics in metrics['features'].items():
        features_data.append({
            'Feature': feature_name,
            'MAE': feature_metrics['mae'],
            'MSE': feature_metrics['mse'],
            'MAPE': feature_metrics['mape'],
            'R2': feature_metrics['r2']
        })
    features_df = pd.DataFrame(features_data)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        overall_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
        features_df.to_excel(writer, sheet_name='Feature-wise Metrics', index=False)


def train_model(model: nn.Module, train_data: Data, val_data: Data, test_data: Data,
                train_target: torch.Tensor, val_target: torch.Tensor, test_target: torch.Tensor,
                epochs: int = 300, lr: float = 0.001, weight_decay: float = 1e-3, 
                warmup_epochs: int = 50, min_lr: float = 1e-6, patience: int = 20,
                save_dir: str = './results',
                rate_penalty_weight: float = 0.1) -> tuple:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, min_lr=min_lr, max_lr=lr)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_out_predictions, train_reaction_rates = model(train_data.x, train_data.edge_index)
        train_loss = mse_criterion(train_out_predictions, train_target)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_out_predictions, val_reaction_rates = model(val_data.x, val_data.edge_index)
            if val_out_predictions.shape[1] != val_target.shape[1]:
                val_out_predictions = val_out_predictions[:, :val_target.shape[1], :]
            val_loss = mse_criterion(val_out_predictions, val_target)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
    model.load_state_dict(best_model_state)
    final_test_metrics = evaluate_model(model, test_data, test_target)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_metrics = evaluate_model(model, train_data, train_target)
    val_metrics = evaluate_model(model, val_data, val_target)
    test_metrics = evaluate_model(model, test_data, test_target)
    save_metrics_to_excel(train_metrics, os.path.join(save_dir, f'train_metrics_{timestamp}.xlsx'))
    save_metrics_to_excel(val_metrics, os.path.join(save_dir, f'val_metrics_{timestamp}.xlsx'))
    save_metrics_to_excel(test_metrics, os.path.join(save_dir, f'test_metrics_{timestamp}.xlsx'))
    print(f"All evaluation results saved to {save_dir}")
    return train_losses, val_losses, final_test_metrics
