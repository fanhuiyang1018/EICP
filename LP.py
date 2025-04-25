#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   LP.py
@Time    :   2025/04/25 15:41:49
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import pickle
import shap
from multiprocessing import Pool, cpu_count
import os

# Set matplotlib parameters for better visualization
plt.rcParams.update({
    'font.family': 'Times New Roman',  
    'font.size': 21.5,                 
    'axes.linewidth': 2,               
    'xtick.major.width': 2,           
    'ytick.major.width': 2,           
    'xtick.major.size': 8,            
    'ytick.major.size': 8,            
    'xtick.direction': 'in',          
    'ytick.direction': 'in'           
})

def process_data(input_file, lasso_features_file):
    """
    Load and process the data, extract features and target variable, and perform standardization.
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
    train_data, non_train_data = train_test_split(data, test_size=0.2, random_state=1142)  
    val_data, test_data = train_test_split(non_train_data, test_size=0.5, random_state=1142)  

    X_train = train_data[lasso_features]
    y_train = train_data['k']

    X_val = val_data[lasso_features]
    y_val = val_data['k']

    X_test = test_data[lasso_features]
    y_test = test_data['k']
    temperature_test = test_data['T']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    temperature_test_tensor = torch.tensor(temperature_test.values, dtype=torch.float32)

    return (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            X_test_tensor, y_test_tensor, temperature_test_tensor, scaler, lasso_features, lasso_weights)

class ElectrolyteModel(nn.Module):
    """
    Define the neural network structure for the electrolyte model.
    """
    def __init__(self, input_dim):
        super(ElectrolyteModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 140)
        self.fc2 = nn.Linear(140, 140)
        self.fc3 = nn.Linear(140, 140)
        self.fc4 = nn.Linear(140, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def train_model(model, X_train, y_train, X_val, y_val, epochs=300, patience=10):
    """
    Train the model with early stopping to prevent overfitting.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}")
                break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.2f}, Val Loss: {val_loss.item():.2f}')
    return train_losses, val_losses

def calculate_shap_values_for_chunk(chunk, model, X_train_tensor):
    """
    Calculate SHAP values for a single chunk of data.
    """
    chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
    device = next(model.parameters()).device
    chunk_tensor = chunk_tensor.to(device)
    X_train_tensor = X_train_tensor.to(device)
    
    explainer = shap.GradientExplainer(model, X_train_tensor)
    shap_values = explainer.shap_values(chunk_tensor)
    return shap_values

def calculate_shap_values_parallel(X_train, X_test, model, features, num_processes=None):
    """
    Calculate SHAP values in parallel.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    if num_processes is None:
        num_processes = cpu_count()

    pool = Pool(processes=num_processes)
    chunk_size = len(X_test_tensor) // num_processes
    chunks = [X_test_tensor[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes - 1)]
    chunks.append(X_test_tensor[(num_processes - 1) * chunk_size:])

    shap_values_list = pool.starmap(calculate_shap_values_for_chunk, [(chunk.numpy(), model, X_train_tensor) for chunk in chunks])
    pool.close()
    pool.join()

    shap_values = np.vstack(shap_values_list)  # Stack SHAP values vertically
    return shap_values

def plot_all_shap_values(shap_values, X_test, feature_names, output_dir):
    """
    Plot SHAP values for all features.
    """
    # Convert shap_values from (n_samples, n_features, 1) to (n_samples, n_features)
    shap_values_2d = np.squeeze(shap_values, axis=-1)

    # Plot SHAP values for all features (dot plot)
    shap.summary_plot(shap_values_2d, X_test, feature_names=feature_names, plot_type="dot", max_display=len(feature_names), show=False)
    plt.savefig(os.path.join(output_dir, "all_shap_values_dot.png"), bbox_inches="tight")
    plt.close()

    # Plot average SHAP values (bar plot)
    shap.summary_plot(shap_values_2d, X_test, feature_names=feature_names, plot_type="bar", max_display=len(feature_names), show=False)
    plt.xlabel('Average SHAP value')
    plt.savefig(os.path.join(output_dir, "average_shap_values_bar.png"), bbox_inches="tight")
    plt.close()

def plot_results(train_losses, val_losses, y_test_tensor, predictions, temperature_test_tensor, MSLF_test, output_dir):
    """
    Plot training and validation losses, and compare predicted results with experimental results.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, 'r-', label='Training Loss', linewidth=3)  
    plt.plot(val_losses, 'b-', label='Validation Loss', linewidth=3)  
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('MSELoss', fontweight='bold')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir, 'loss_vs_epoch.png'), bbox_inches='tight')  
    plt.close()  

    plt.figure(figsize=(8, 6))
    rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), predictions.numpy()))
    r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())
    plt.scatter(y_test_tensor.numpy(), predictions.numpy(), alpha=0.5, label=f'RMSE: {rmse:.2f}, R²: {r2:.2f}', s=80, color='#0072C6', linewidth=1)  
    plt.plot([y_test_tensor.min(), y_test_tensor.max()], [y_test_tensor.min(), y_test_tensor.max()], 'k--', lw=3)  
    plt.xlabel('Experimental Ionic Conductivity (mS/cm)', fontweight='bold')
    plt.ylabel('Predicted Ionic Conductivity (mS/cm)', fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower right', frameon=False, prop={'weight': 'bold'}) 
    plt.savefig(os.path.join(output_dir, 'predicted_vs_experimental.png'), bbox_inches='tight')  
    plt.close()  

    plt.figure(figsize=(8, 6))
    if temperature_test_tensor.isnan().any():
        raise ValueError("Temperature data contains NaN values")
        # Check for zero values in temperature data
    if (temperature_test_tensor == 0).any():
        raise ValueError("Temperature data contains zero values")
    plt.scatter(1000 / temperature_test_tensor.numpy(), y_test_tensor.numpy(), label='Experimental', color='black', s=80)  
    plt.scatter(1000 / temperature_test_tensor.numpy(), predictions.numpy(), label='Predicted', color='red', alpha=0.5, s=80)  
    plt.xlabel('1000/Temperature (K⁻¹)', fontweight='bold')
    plt.ylabel('Ionic Conductivity (mS/cm)', fontweight='bold')
    plt.legend(frameon=False)
    plt.xscale('linear')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'ionic_conductivity_vs_temperature.png'), bbox_inches='tight')  
    plt.close()  
    plt.figure(figsize=(8, 6))
    plt.scatter(MSLF_test, y_test_tensor.numpy(), alpha=0.5, color='#0072C6', s=80)
    plt.xlabel('MSLF', fontweight='bold')
    plt.ylabel('Ionic Conductivity (mS/cm)', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'mslf_vs_experimental.png'), bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python LP.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv lasso_features_P.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    lasso_features_file = sys.argv[2]

    # Get the directory of the input file
    input_dir = os.path.dirname(input_file)
    figures_dir = os.path.join(input_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, temperature_test_tensor, scaler, lasso_features, lasso_weights = process_data(input_file, lasso_features_file)
    model = ElectrolyteModel(X_train_tensor.shape[1])
    train_losses, val_losses = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = nn.MSELoss()(predictions, y_test_tensor)

    rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), predictions.numpy()))
    r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())
    print(f'Test Loss: {test_loss.item():.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R2: {r2:.2f}')

    X_test_numpy = X_test_tensor.numpy()
    MSLF_test = np.sum(X_test_numpy * lasso_weights, axis=1)

    plot_results(train_losses, val_losses, y_test_tensor, predictions, temperature_test_tensor, MSLF_test, figures_dir)
    torch.save(model.state_dict(), os.path.join('model.pth'))
    with open(os.path.join('scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Calculate and plot SHAP values
    shap_values = calculate_shap_values_parallel(X_train_tensor.numpy(), X_test_numpy, model, lasso_features)
    plot_all_shap_values(shap_values, X_test_numpy, lasso_features, figures_dir)

if __name__ == "__main__":
    main()