#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   RdKit_LASSO_D.py
@Time    :   2025/04/25 15:42:02
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import sys
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

def load_smiles_dict(smiles_dict_file):
    """
    Load a dictionary of SMILES strings from a text file.
    
    The file should contain a Python dictionary-like structure, e.g., {"key": "value"}.
    If the file content is not properly formatted as a dictionary, this function will attempt to correct it.
    
    Args:
        smiles_dict_file (str): Path to the file containing the SMILES dictionary.
    
    Returns:
        dict: A dictionary where keys are compound names and values are their corresponding SMILES strings.
    """
    with open(smiles_dict_file, "r") as f:
        content = f.read()
        # Ensure the content is properly formatted as a dictionary
        if not content.startswith("{"):
            content = "{" + content
        if not content.endswith("}"):
            content = content + "}"
        smiles_dict = eval(content)
    return smiles_dict

def calculate_descriptors(smiles_dict, desc_list):
    """
    Calculate molecular descriptors for a given dictionary of SMILES strings.
    
    Args:
        smiles_dict (dict): Dictionary with compound names as keys and SMILES strings as values.
        desc_list (list): List of descriptor names to calculate.
    
    Returns:
        dict: A nested dictionary where each key is a compound name and each value is another dictionary
              containing the calculated descriptors.
    """
    mol_descriptors = {}
    for col, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_descriptors[col] = {desc_name: getattr(Descriptors, desc_name)(mol) for desc_name in desc_list}
    return mol_descriptors

def calculate_row(idx, row, mol_descriptors, salt_descriptors, scalers, desc_list):
    """
    Calculate weighted mean and variance of descriptors for a single row of data.
    
    Args:
        idx (int): Index of the row.
        row (pd.Series): A row from the DataFrame.
        mol_descriptors (dict): Descriptors for molecules.
        salt_descriptors (dict): Descriptors for salts.
        scalers (dict): Scalers for normalizing descriptor values.
        desc_list (list): List of descriptor names.
    
    Returns:
        tuple: A tuple containing the row index and a dictionary of calculated descriptors.
    """
    row_results = {}
    for desc_name in desc_list:
        values = []  # Store descriptor values
        weights = []  # Store weights
        # Calculate weighted mean for molecular descriptors
        for col in mol_descriptors.keys():
            if col in row and row[col] > 0:
                raw_value = mol_descriptors[col][desc_name]
                scaled_value = scalers[desc_name].transform([[raw_value]])[0][0]
                values.append(scaled_value)
                weights.append(row[col])

        if values and weights:
            # Calculate weighted mean
            weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            # Calculate weighted variance
            weighted_variance = sum(w * (v - weighted_mean) ** 2 for v, w in zip(values, weights)) / sum(weights)
        else:
            weighted_mean = None
            weighted_variance = None

        # Add salt descriptor values if applicable
        for col in salt_descriptors.keys():
            if col in row and row[col] > 0:
                raw_value = salt_descriptors[col][desc_name]
                scaled_value = scalers[desc_name].transform([[raw_value]])[0][0]
                if weighted_mean is not None:
                    weighted_mean += scaled_value
                break

        row_results[f"{desc_name}_mean"] = weighted_mean
        row_results[f"{desc_name}_variance"] = weighted_variance
    return idx, row_results

def process_data(input_file, smiles_dict_1, smiles_dict_2):
    """
    Process the input data by calculating and adding molecular descriptors.
    
    Args:
        input_file (str): Path to the input CSV file.
        smiles_dict_1 (dict): Dictionary of SMILES strings for molecules.
        smiles_dict_2 (dict): Dictionary of SMILES strings for salts.
    """
    # Load input data
    data = pd.read_csv(input_file)
    # Get all available descriptors and filter out unwanted ones
    all_desc_list = [x[0] for x in Descriptors._descList]
    excluded_desc = [
        "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
        "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"
    ]
    desc_list = [desc for desc in all_desc_list if desc not in excluded_desc]

    # Initialize the descriptor DataFrame with only the filtered descriptors' mean and variance columns
    descriptors_df = pd.DataFrame(index=data.index, columns=[f"{desc}_mean" for desc in desc_list] + [f"{desc}_variance" for desc in desc_list])

    # Pre-calculate descriptors for all molecules
    mol_descriptors = calculate_descriptors(smiles_dict_1, desc_list)
    salt_descriptors = calculate_descriptors(smiles_dict_2, desc_list)

    # Collect all descriptor values for scaling
    all_values = {desc_name: [] for desc_name in desc_list}
    for col, desc_dict in mol_descriptors.items():
        for desc_name, value in desc_dict.items():
            all_values[desc_name].append(value)
    for col, desc_dict in salt_descriptors.items():
        for desc_name, value in desc_dict.items():
            all_values[desc_name].append(value)

    # Fit scalers for each descriptor
    scalers = {desc_name: StandardScaler().fit([[v] for v in values]) for desc_name, values in all_values.items()}

    # Use parallel processing to calculate descriptors for each row
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_row, idx, row, mol_descriptors, salt_descriptors, scalers, desc_list)
            for idx, row in data.iterrows()
        ]
        for future in futures:
            idx, row_results = future.result()
            for col, value in row_results.items():
                descriptors_df.at[idx, col] = value

    # Combine the original data with the calculated descriptors
    data_with_descriptors = pd.concat([data, descriptors_df], axis=1)

    # Save the output file
    output_file = input_file.replace(".csv", "_with_descriptors.csv")
    data_with_descriptors.to_csv(output_file, index=False)

def main():
    """
    Main program entry point.
    Get the input file path and SMILES dictionary file path from the command line arguments.
    If the number of command line arguments is incorrect, print usage instructions and exit the program.
    """
    if len(sys.argv) != 3:
        print("Usage: python RdKit_LASSO_D.py CALiSol-23_Dataset_cleaned_data.csv SMILES_dict.txt")
        sys.exit(1)

    # Get the input file path and SMILES dictionary file path
    input_file = sys.argv[1]
    smiles_dict_file = sys.argv[2]

    # Load SMILES dictionaries
    smiles_dict = load_smiles_dict(smiles_dict_file)
    smiles_dict_1 = {k: v for k, v in smiles_dict.items() if not k.startswith("Li")}  # Molecules
    smiles_dict_2 = {k: v for k, v in smiles_dict.items() if k.startswith("Li")}  # Salts

    # Process the data and add descriptors
    process_data(input_file, smiles_dict_1, smiles_dict_2)

if __name__ == "__main__":
    main()