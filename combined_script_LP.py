#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   combined_script_LP.py
@Time    :   2025/04/25 15:41:31
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import os
import subprocess
import pandas as pd
import torch
import pickle
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import sys

# Set plotting style
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'axes.labelweight': 'normal',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 2
})

# Ensure the output directory exists
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Define SMILES dictionary file
smiles_dict_file = "SMILES_dict.txt"

# Define model and scaler files
model_path = "model.pth"
scaler_path = "scaler.pkl"
lasso_features_file = "lasso_features_P.txt"

# Define solvents with their molar masses (g/mol) and densities (g/mL)
solvents = {
    'EC': (88.06, 1.3210), 'PC': (102.08, 1.205), 'DMC': (90.08, 1.07),
    'EMC': (104.10, 0.902), 'DEC': (118.132, 0.975), 'DME': (90.12, 0.86),
    'DMSO': (78.13, 1.1004), 'AN': (41.05, 0.786), 'MOEMC': (134.13, 1.5),
    'TTFP': (344.07, 1.487), 'EA': (88.10, 0.902), 'MA': (74.08, 0.932),
    'FEC': (106.05, 1.454), 'DOL': (74.08, 1.06), '2-MeTHF': (86.13, 0.854),
    'DMM': (162.2, 0.902), 'Freon 11': (137.36, 1.49), 'Methylene chloride': (84.93, 1.33),
    'THF': (72.10, 0.888), 'Toluene': (92.14, 0.867), 'Sulfolane': (120.17, 1.26),
    '2-Glyme': (134.17, 0.937), '3-Glyme': (178.22, 0.986), '4-Glyme': (222.28, 1.009),
    '3-Me-2-Oxazolidinone': (101.10, 1.17), '3-MeSulfolane': (134.20, 1.20),
    'Ethyldiglyme': (134.17, 0.937), 'DMF': (73.09, 0.944), 'Ethylbenzene': (106.17, 0.866),
    'Ethylmonoglyme': (76.10, 0.965), 'Benzene': (78.11, 0.876), 'g-Butyrolactone': (86.09, 1.13),
    'Cumene': (120.19, 0.862), 'Propylsulfone': (150.24, 1.109), 'Pseudocumeme': (120.19, 0.876),
    'TEOS': (208.33, 0.940), 'm-Xylene': (106.17, 0.860), 'o-Xylene': (106.16, 0.87596)
}

# Load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully.")
        return model_state_dict, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        sys.exit(1)

# Load LASSO features
def load_lasso_features(lasso_features_file):
    try:
        with open(lasso_features_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        lasso_features = [line.split(":")[0].strip() for line in lines]
        lasso_weights = np.array([float(line.split(":")[1].strip()) for line in lines])
        return lasso_features, lasso_weights
    except Exception as e:
        print(f"Error loading Lasso features: {e}")
        sys.exit(1)

# Preprocess data
def preprocess_data(data, scaler, lasso_features):
    if data.empty:
        raise ValueError("Input data is empty. Please check the input file or data cleaning process.")
    
    missing_features = set(lasso_features) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    X = data[lasso_features]
    if X.empty:
        raise ValueError("Feature data is empty. Please check the feature extraction process.")
    
    return torch.tensor(scaler.transform(X), dtype=torch.float32)

# Define the model
class ElectrolyteModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 140)
        self.fc2 = torch.nn.Linear(140, 140)
        self.fc3 = torch.nn.Linear(140, 140)
        self.fc4 = torch.nn.Linear(140, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x)

# Plot results
def plot_results(data, output_prefix):
    data = data.sort_values(by='T')
    inv_T = 1000 / data['T']
    plt.figure(figsize=(8, 6))
    plt.scatter(inv_T, data['k'], label='Experimental', color='blue', marker='o', s=100)
    plt.scatter(inv_T, data['predicted_k'], label='Predicted X_LP_S', color='red', marker='^', s=100)
    plt.xlabel('1000/Temperature (K⁻¹)', fontweight='bold')
    plt.ylabel('Ionic Conductivity (mS/cm)', fontweight='bold')
    plt.legend(frameon=False, loc='upper right')
    plt.savefig(output_prefix, bbox_inches='tight')
    plt.close('all')

# Run data cleaning script
def run_clean_data(input_file, output_file):
    subprocess.run(["python", "clean_and_analyze_data.py", input_file], check=True)
    cleaned_file = input_file.replace(".csv", "_cleaned_data.csv")
    if os.path.exists(cleaned_file):
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(cleaned_file, output_file)
    else:
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_file}")

# Run RDKit script to calculate descriptors
def run_rdkit(input_file, smiles_dict_file, output_file):
    subprocess.run(["python", "RdKit_LP.py", input_file, smiles_dict_file], check=True)
    descriptors_file = input_file.replace(".csv", "_with_descriptors.csv")
    if os.path.exists(descriptors_file):
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(descriptors_file, output_file)
    else:
        raise FileNotFoundError(f"Descriptors file not found: {descriptors_file}")

# Process a single file
def process_file(input_file, output_folder):
    cleaned_file = os.path.join(output_folder, os.path.basename(input_file).replace(".csv", "_cleaned.csv"))
    descriptors_file = os.path.join(output_folder, os.path.basename(input_file).replace(".csv", "_with_descriptors.csv"))
    output_file = os.path.join(output_folder, os.path.basename(input_file).replace(".csv", "_output.csv"))
    output_plot = os.path.join(output_folder, os.path.basename(input_file).replace(".csv", "_output.png"))

    print(f"Processing file: {input_file}")  # Debug information

    # Load input file and extract information
    input_data = pd.read_csv(input_file)
    print(f"Loaded data shape: {input_data.shape}")  # Debug information

    if input_data.empty:
        print(f"Input file is empty: {input_file}. Skipping...")
        return None

    input_file_name = os.path.basename(input_file)
    doi = input_data['doi'].iloc[0] if 'doi' in input_data.columns else 'N/A'
    salt_name = input_data['salt'].iloc[0] if 'salt' in input_data.columns else 'Unknown'

    # Run data cleaning script
    run_clean_data(input_file, cleaned_file)

    # Run RDKit script to calculate descriptors
    run_rdkit(cleaned_file, smiles_dict_file, descriptors_file)

    # Load data with descriptors
    data_with_descriptors = pd.read_csv(descriptors_file)
    print(f"Data with descriptors shape: {data_with_descriptors.shape}")  # Debug information

    if data_with_descriptors.empty:
        print(f"Data with descriptors is empty: {descriptors_file}. Skipping...")
        return None

    # Preprocess data
    try:
        X_tensor = preprocess_data(data_with_descriptors, scaler, lasso_features)
    except Exception as e:
        print(f"Error during preprocessing: {e}. File: {input_file}")
        return None

    # Load model
    model = ElectrolyteModel(len(lasso_features))
    model.load_state_dict(model_state_dict)
    model.eval()

    # Predict
    with torch.no_grad():
        data_with_descriptors['predicted_k'] = model(X_tensor).numpy().ravel()

    # Calculate MSLF
    exclude_features = ['T', 'c']
    lasso_features_filtered = [feature for feature in lasso_features if feature not in exclude_features]
    lasso_weights_filtered = np.array([lasso_weights[i] for i, feature in enumerate(lasso_features) if feature not in exclude_features])

    X_scaled = scaler.transform(data_with_descriptors[lasso_features])
    X_scaled_filtered = X_scaled[:, [i for i, feature in enumerate(lasso_features) if feature not in exclude_features]]
    MSLF = np.sum(X_scaled_filtered[0] * lasso_weights_filtered)

    # Save results
    data_with_descriptors.to_csv(output_file, index=False)
    plot_results(data_with_descriptors, output_plot)

    # Extract k values around 298.15 K
    target_T = 298.15
    closest_temp = data_with_descriptors['T'].min()

    if abs(closest_temp - target_T) < 1e-5:
        k_value_exp = data_with_descriptors.loc[data_with_descriptors['T'].idxmin(), 'k']
        k_value_pred = data_with_descriptors.loc[data_with_descriptors['T'].idxmin(), 'predicted_k']
    else:
        lower_temp_row = data_with_descriptors[data_with_descriptors['T'] < target_T]
        upper_temp_row = data_with_descriptors[data_with_descriptors['T'] > target_T]

        if not lower_temp_row.empty:
            lower_temp_row = lower_temp_row.iloc[-1]
        else:
            lower_temp_row = None

        if not upper_temp_row.empty:
            upper_temp_row = upper_temp_row.iloc[0]
        else:
            upper_temp_row = None

        if lower_temp_row is None and upper_temp_row is None:
            print(f"No data points found around target temperature {target_T} in file {input_file}. Skipping...")
            return None

        if lower_temp_row is None:
            k_value_exp = upper_temp_row['k']
            k_value_pred = upper_temp_row['predicted_k']
        elif upper_temp_row is None:
            k_value_exp = lower_temp_row['k']
            k_value_pred = lower_temp_row['predicted_k']
        else:
            k_value_exp = lower_temp_row['k'] + (upper_temp_row['k'] - lower_temp_row['k']) * ((target_T - lower_temp_row['T']) / (upper_temp_row['T'] - lower_temp_row['T']))
            k_value_pred = lower_temp_row['predicted_k'] + (upper_temp_row['predicted_k'] - lower_temp_row['predicted_k']) * ((target_T - lower_temp_row['T']) / (upper_temp_row['T'] - lower_temp_row['T']))

    # Summarize results
    result = {
        'input_file': input_file_name,
        'doi': doi,
        'salt': salt_name,
        'solvent_info': ', '.join([f"{solvent}: {data_with_descriptors[solvent].iloc[0]:.2f}" for solvent in solvents if solvent in data_with_descriptors.columns and data_with_descriptors[solvent].iloc[0] > 0]),
        'k_value_exp': round(k_value_exp, 3),
        'k_value_pred': round(k_value_pred, 3),
        'MSLF': round(MSLF, 3)
    }
    return result

# Main program
if __name__ == "__main__":
    model_state_dict, scaler = load_model_and_scaler(model_path, scaler_path)
    lasso_features, lasso_weights = load_lasso_features(lasso_features_file)

    all_results = []

    # Read all files from the 'groups' subdirectory
    groups_folder = "groups"
    if not os.path.exists(groups_folder):
        print(f"Directory not found: {groups_folder}")
        sys.exit(1)

    for file_name in os.listdir(groups_folder):
        if file_name.endswith(".csv"):
            input_file = os.path.join(groups_folder, file_name)
            result = process_file(input_file, output_folder)
            if result is not None:
                all_results.append(result)

    # Save all results to a single CSV file
    summary_file = os.path.join(output_folder, "summary.csv")
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_file, index=False)