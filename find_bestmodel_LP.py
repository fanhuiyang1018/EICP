#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   find_bestmodel_LP.py
@Time    :   2025/04/25 15:41:40
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os

# Set matplotlib parameters for better visualization
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'black'

def process_data(input_file, lasso_features_file):
    """
    Process the input data by loading LASSO features and splitting the data into training, validation, and test sets.
    
    Args:
        input_file (str): Path to the input CSV file.
        lasso_features_file (str): Path to the file containing LASSO feature names and weights.
    
    Returns:
        tuple: A tuple containing the processed data tensors and the scaler object.
    """
    data = pd.read_csv(input_file)
    with open(lasso_features_file, 'r') as f:
        lasso_features = [line.strip() for line in f.readlines()]
    
    lasso_weights = {}
    for line in lasso_features:
        feature, weight = line.split(": ")
        lasso_weights[feature.strip()] = float(weight.strip())
    
    lasso_features = list(lasso_weights.keys())
    lasso_weights = np.array(list(lasso_weights.values()))

    y = data['k']
    X = data[lasso_features]

    # Split the data into training, validation, and test sets
    train_data, non_train_data = train_test_split(data, test_size=0.2, random_state=1142)
    val_data, test_data = train_test_split(non_train_data, test_size=0.5, random_state=1142)

    X_train = train_data[lasso_features]
    y_train = train_data['k']

    X_val = val_data[lasso_features]
    y_val = val_data['k']

    X_test = test_data[lasso_features]
    y_test = test_data['k']
    temperature_test = test_data['T']

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    temperature_test_tensor = torch.tensor(temperature_test.values, dtype=torch.float32)

    return (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            X_test_tensor, y_test_tensor, temperature_test_tensor, scaler, lasso_features)

class ElectrolyteModel(nn.Module):
    """
    A neural network model for predicting ionic conductivity.
    """
    def __init__(self, input_dim, hidden_layers, neurons_per_layer):
        super(ElectrolyteModel, self).__init__()
        layers = [nn.Linear(input_dim, neurons_per_layer), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, X_train, y_train, X_val, y_val, epochs=300, patience=10):
    """
    Train the neural network model with early stopping.
    
    Args:
        model (nn.Module): The neural network model.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training target.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation target.
        epochs (int, optional): Number of epochs. Defaults to 300.
        patience (int, optional): Patience for early stopping. Defaults to 10.
    
    Returns:
        float: Best validation loss.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}")
                break

    return best_val_loss

def save_heatmaps(rmse_matrix, time_matrix, hidden_layers_options, neurons_per_layer_options, output_dir):
    """
    Save RMSE and training time heatmaps to the specified directory.
    
    Args:
        rmse_matrix (np.ndarray): Matrix of RMSE values.
        time_matrix (np.ndarray): Matrix of training times.
        hidden_layers_options (list): List of hidden layer options.
        neurons_per_layer_options (list): List of neuron options per layer.
        output_dir (str): Directory to save the heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(rmse_matrix, annot=True, fmt=".3f", cmap="coolwarm", 
                xticklabels=hidden_layers_options, yticklabels=neurons_per_layer_options, 
                edgecolor="none", linewidths=0)
    plt.xlabel('Hidden Layers', fontweight='bold')
    plt.ylabel('Neurons per Layer', fontweight='bold')
    plt.title('RMSE of X_LP_S', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_heatmap.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(time_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=hidden_layers_options, yticklabels=neurons_per_layer_options, 
                edgecolor="none", linewidths=0)
    plt.xlabel('Hidden Layers', fontweight='bold')
    plt.ylabel('Neurons per Layer', fontweight='bold')
    plt.title('Training Times of X_LP_S (s)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_heatmap.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python find_bestmodel_LP.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv lasso_features_P.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    lasso_features_file = sys.argv[2]
    output_dir = "figures"  # Fixed output directory

    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, temperature_test_tensor, scaler, lasso_features = process_data(input_file, lasso_features_file)

    hidden_layers_options = [1, 2, 3, 4]
    neurons_per_layer_options = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    rmse_matrix = np.zeros((len(neurons_per_layer_options), len(hidden_layers_options)))
    time_matrix = np.zeros((len(neurons_per_layer_options), len(hidden_layers_options)))
    num_runs = 3

    for i, neurons_per_layer in enumerate(neurons_per_layer_options):
        for j, hidden_layers in enumerate(hidden_layers_options):
            print(f"Training model with {hidden_layers} hidden layers and {neurons_per_layer} neurons per layer")
            rmse_sum = 0
            time_sum = 0

            for run in range(num_runs):
                model = ElectrolyteModel(X_train_tensor.shape[1], hidden_layers, neurons_per_layer)
                start_time = time.time()
                best_val_loss = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
                end_time = time.time()
                training_time = end_time - start_time

                model.eval()
                with torch.no_grad():
                    predictions = model(X_test_tensor)
                    test_loss = nn.MSELoss()(predictions, y_test_tensor)
                    rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), predictions.numpy()))
                    rmse_sum += rmse
                    time_sum += training_time

            rmse_matrix[i, j] = rmse_sum / num_runs
            time_matrix[i, j] = time_sum / num_runs

            print(f'Average Test RMSE: {rmse_matrix[i, j]:.4f}, Average Training Time: {time_matrix[i, j]:.2f}s, Config: {hidden_layers} layers, {neurons_per_layer} neurons')

    save_heatmaps(rmse_matrix, time_matrix, hidden_layers_options, neurons_per_layer_options, output_dir)

if __name__ == "__main__":
    main()