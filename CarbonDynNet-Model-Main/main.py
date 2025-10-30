import torch
from model import SewerGNN
from data_utils import load_node_data
from train import split_data, train_model

if __name__ == "__main__":
    data_dir = "./your_data_dir_here"  # Set to your data path
    save_dir = "./results"
    train_ratio = 0.6
    val_ratio = 0.2
    hidden_dim = 256
    dropout_rate = 0.5
    epochs = 300
    lr = 0.001
    weight_decay = 1e-3
    warmup_epochs = 100
    min_lr = 1e-6
    patience = 20
    rate_penalty_weight = 0.01
    print("Loading data...")
    node_features, target_features, processed_files, node_mapping = load_node_data(data_dir)
    num_nodes, time_steps, num_features = node_features.shape
    input_dim = num_features
    output_dim = None
    print("Splitting data...")
    train_data, val_data, test_data, train_target, val_target, test_target = split_data(
        node_features, target_features, train_ratio, val_ratio
    )
    print("Building model...")
    model = SewerGNN(input_dim, hidden_dim, output_dim, dropout_rate=dropout_rate, num_nodes=num_nodes)
    print("Training model...")
    train_losses, val_losses, test_metrics = train_model(
        model, train_data, val_data, test_data,
        train_target, val_target, test_target,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        warmup_epochs=warmup_epochs, min_lr=min_lr, patience=patience,
        save_dir=save_dir, rate_penalty_weight=rate_penalty_weight
    )
    print("Final test metrics:")
    print(test_metrics)
