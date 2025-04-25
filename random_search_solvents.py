#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   random_search_solvents.py
@Time    :   2025/04/25 16:00:00
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import subprocess
import sys
from combined_script_LP import ElectrolyteModel, preprocess_data, load_model_and_scaler, solvents
from clean_and_analyze_data import clean_data  # Import the data cleaning function

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def load_lasso_features(lasso_features_file):
    """
    Load LASSO features and weights from the specified file.
    
    Args:
        lasso_features_file (str): Path to the LASSO features file.
    
    Returns:
        tuple: List of LASSO features and their corresponding weights.
    """
    try:
        with open(lasso_features_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        lasso_features = [line.split(":")[0].strip() for line in lines]
        lasso_weights = np.array([float(line.split(":")[1].strip()) for line in lines])
        return lasso_features, lasso_weights
    except Exception as e:
        print(f"Error loading Lasso features: {e}")
        raise

def generate_random_solvent_ratios(solvents, max_solvents=5):
    """
    Randomly generate solvent ratios, ensuring the sum of ratios is 1 and the number of solvents does not exceed 5.
    
    Args:
        solvents (list): List of available solvents.
        max_solvents (int): Maximum number of solvents, default is 5.
    
    Returns:
        dict: Dictionary of solvents and their corresponding ratios.
    """
    if isinstance(solvents, dict):
        solvents = list(solvents.keys())
    elif isinstance(solvents, set):
        solvents = list(solvents)

    num_solvents = random.randint(1, min(len(solvents), max_solvents))
    selected_solvents = random.sample(solvents, num_solvents)
    ratios = np.random.rand(num_solvents)
    ratios /= ratios.sum()  # Normalize to ensure the sum of ratios is 1
    return dict(zip(selected_solvents, ratios))

def evaluate_solvent_ratios(ratios, model, scaler, lasso_features, data):
    """
    Evaluate the ionic conductivity for given solvent ratios.
    
    Args:
        ratios (dict): Dictionary of solvents and their corresponding ratios.
        model (ElectrolyteModel): Trained model.
        scaler (StandardScaler): Data scaler.
        lasso_features (list): List of LASSO features.
        data (pd.DataFrame): Input data.
    
    Returns:
        float: Predicted ionic conductivity (k).
    """
    # Add solvent ratios to the data
    for solvent, ratio in ratios.items():
        data[solvent] = ratio

    # Preprocess data
    X_tensor = preprocess_data(data, scaler, lasso_features)

    # Model prediction
    model.eval()
    with torch.no_grad():
        predicted_k = model(X_tensor).numpy().ravel()[0]
    
    return predicted_k

def random_search(input_file, max_iterations=10, max_solvents=5):
    """
    Perform a random search for the best solvent ratios.
    
    Args:
        input_file (str): Path to the input data file.
        max_iterations (int): Maximum number of iterations.
        max_solvents (int): Maximum number of solvents.
    
    Returns:
        dict: Best solvent ratios and their corresponding ionic conductivity.
    """
    model_path = "model.pth"
    scaler_path = "scaler.pkl"
    lasso_features_file = "lasso_features_P.txt"

    # Load model, scaler, and LASSO features
    model_state_dict, scaler = load_model_and_scaler(model_path, scaler_path)
    lasso_features, _ = load_lasso_features(lasso_features_file)
    model = ElectrolyteModel(len(lasso_features))
    model.load_state_dict(model_state_dict)

    # Load input data
    data = pd.read_csv(input_file)

    best_ratios = None
    best_k = -np.inf

    for i in range(max_iterations):
        ratios = generate_random_solvent_ratios(solvents, max_solvents)
        
        # Create a temporary data file
        temp_data_file = "temp_data.csv"
        data_with_ratios = data.copy()
        for solvent, ratio in ratios.items():
            data_with_ratios[solvent] = ratio
        data_with_ratios.to_csv(temp_data_file, index=False)

        # Data cleaning and feature extraction
        cleaned_file = temp_data_file.replace(".csv", "_cleaned_data.csv")
        descriptor_file = cleaned_file.replace(".csv", "_with_descriptors.csv")

        # Call the data cleaning script
        subprocess.run(["python", "clean_and_analyze_data.py", temp_data_file], check=True)
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned file not found: {cleaned_file}")

        # Call the feature extraction script
        subprocess.run(["python", "RdKit_LP.py", cleaned_file, "SMILES_dict.txt"], check=True)
        if not os.path.exists(descriptor_file):
            raise FileNotFoundError(f"Descriptor file not found: {descriptor_file}")

        # Load the data with descriptors
        data_with_descriptors = pd.read_csv(descriptor_file)

        # Evaluate ionic conductivity
        k = evaluate_solvent_ratios(ratios, model, scaler, lasso_features, data_with_descriptors)
        
        if k > best_k:
            best_k = k
            best_ratios = ratios
            print(f"Iteration {i+1}: Found better ratios with k = {best_k:.4f}")
            print(f"Best ratios: {best_ratios}")

        # Remove temporary files
        os.remove(temp_data_file)
        os.remove(cleaned_file)
        os.remove(descriptor_file)

    return best_ratios, best_k

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python random_search_solvents.py example_data.csv 50")
        sys.exit(1)

    input_file = sys.argv[1]
    max_iterations = int(sys.argv[2])
    max_solvents = 5  # Default value

    best_ratios, best_k = random_search(input_file, max_iterations, max_solvents)

    # Output the best combination to a text file
    output_file = "best_solvent_ratios.txt"
    with open(output_file, "w") as f:
        f.write(f"Best ion conductivity (k): {best_k:.4f}\n")
        f.write("Best solvent ratios:\n")
        for solvent, ratio in best_ratios.items():
            f.write(f"{solvent}: {ratio:.4f}\n")

    print(f"Best ion conductivity (k): {best_k:.4f}")
    print(f"Best solvent ratios: {best_ratios}")
    print(f"Results saved to {output_file}")