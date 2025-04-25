#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   LASSO_features_order.py
@Time    :   2025/04/25 16:00:00
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os

def load_smiles_dict(smiles_dict_file):
    """
    Load a dictionary of SMILES strings from a text file.
    
    Args:
        smiles_dict_file (str): Path to the file containing the SMILES dictionary.
    
    Returns:
        dict: A dictionary where keys are compound names and values are their corresponding SMILES strings.
    """
    with open(smiles_dict_file, 'r') as f:
        content = f.read()
        # Ensure the content is properly formatted as a dictionary
        if not content.startswith("{"):
            content = "{" + content
        if not content.endswith("}"):
            content = content + "}"
        try:
            smiles_dict = eval(content)
        except Exception as e:
            raise ValueError(f"Error parsing SMILES dictionary file: {e}")
    return smiles_dict

def calculate_descriptors(features_file, smiles_dict_file):
    """
    Main function to calculate descriptors, process input files, and perform the following steps:
    1. Load and clean the feature file.
    2. Load the SMILES dictionary.
    3. Calculate descriptors.
    4. Standardize features.
    5. Save results and plot scatter plots.
    """
    # Load and clean the feature file
    with open(features_file, "r") as f:
        features_to_calculate = [line.strip().split(":")[0].replace("_mean", "").replace("_variance", "") for line in f.readlines() if line.strip()]

    # Ensure features exist in RDKit Descriptors
    valid_features = [feat for feat in features_to_calculate if hasattr(Descriptors, feat)]
    invalid_features = [feat for feat in features_to_calculate if not hasattr(Descriptors, feat)]

    if invalid_features:
        print(f"Warning: The following features are not found in RDKit Descriptors: {', '.join(invalid_features)}")
        features_to_calculate = valid_features

    # Set Matplotlib parameters
    rc = plt.rcParams
    rc['font.family'] = 'Times New Roman'
    size = 28
    rc['font.size'] = rc['axes.labelsize'] = rc['axes.titlesize'] = rc['xtick.labelsize'] = rc['ytick.labelsize'] = size
    rc['axes.labelweight'] = 'normal' 
    rc['xtick.direction'] = rc['ytick.direction'] = 'in'    
    rc['axes.linewidth'] = 2

    # Load SMILES dictionary
    smiles_dict = load_smiles_dict(smiles_dict_file)
    smiles_dict_1 = {k: v for k, v in smiles_dict.items() if not k.startswith('Li')}
    smiles_dict_2 = {k: v for k, v in smiles_dict.items() if k.startswith('Li')}

    # Initialize the descriptor DataFrame
    descriptors_df = pd.DataFrame(columns=['Name', 'Type'] + features_to_calculate)

    data_rows = []
    for name, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            row_data = {'Name': name, 'Type': 'Solvent' if name in smiles_dict_1 else 'Salt'}
            for feat in features_to_calculate:
                try:
                    if feat == 'MaxPartialCharge':
                        AllChem.ComputeGasteigerCharges(mol)
                        partial_charges = [atom.GetDoubleProp(f'_GasteigerCharge') for atom in mol.GetAtoms()]
                        row_data[feat] = max(partial_charges)
                    elif feat == 'MinAbsPartialCharge':
                        AllChem.ComputeGasteigerCharges(mol)
                        partial_charges = [abs(atom.GetDoubleProp(f'_GasteigerCharge')) for atom in mol.GetAtoms()]
                        row_data[feat] = min(partial_charges)
                    else:
                        value = getattr(Descriptors, feat)(mol)
                        row_data[feat] = value
                except Exception as e:
                    print(f"Error calculating {feat} for {name}: {e}")
                    row_data[feat] = None
            data_rows.append(row_data)

    descriptors_df = pd.DataFrame(data_rows)

    # Check for null values in the DataFrame
    if descriptors_df.isnull().values.any():
        print("Warning: Some features contain null values. These will be filled with NaN.")
        descriptors_df.fillna(value=pd.NA, inplace=True)

    # Standardize features
    features_to_scale = descriptors_df.columns[2:]
    if not features_to_scale.empty:
        scaler = StandardScaler()
        descriptors_df[features_to_scale] = scaler.fit_transform(descriptors_df[features_to_scale])
        output_file_scaled = 'solvents_and_salts_descriptors_scaled.csv'
        descriptors_df.to_csv(output_file_scaled, index=False)
    else:
        print("No valid features to scale.")

    # Create a directory to save plots
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Plot scatter plots
    for feature_to_plot in features_to_calculate:
        plt.figure(figsize=(16, 6))
        plt.scatter(descriptors_df['Name'], descriptors_df[feature_to_plot], color='red', marker='o', s=100)
        plt.xticks(rotation=90, fontsize=20)
        plt.title(f'{feature_to_plot}', fontsize=50)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{feature_to_plot}_scatter_plot.png'))
        plt.close()

def main():
    """
    Main program entry point.
    Get the input file paths from the command line arguments.
    If the number of command line arguments is incorrect, print usage instructions and exit the program.
    """
    if len(sys.argv) != 3:
        print("Usage: python LASSO_features_order.py lasso_features.txt SMILES_dict.txt")
        sys.exit(1)

    # Get the input file paths
    features_file = sys.argv[1]
    smiles_dict_file = sys.argv[2]

    # Check if the files exist
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"File {features_file} not found.")
    if not os.path.exists(smiles_dict_file):
        raise FileNotFoundError(f"File {smiles_dict_file} not found.")

    # Call the calculate_descriptors function to start calculating descriptors
    calculate_descriptors(features_file, smiles_dict_file)

if __name__ == "__main__":
    main()