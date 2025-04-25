#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   RdKit_LASSO_S.py
@Time    :   2025/04/25 16:00:00
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

def calculate_descriptors(smiles_dict, desc_list):
    """
    Calculate descriptors for each compound in the SMILES dictionary.
    
    Args:
        smiles_dict (dict): Dictionary with compound names as keys and SMILES strings as values.
        desc_list (list): List of descriptor names to calculate.
    
    Returns:
        dict: A dictionary where each key is a compound name and each value is another dictionary
              containing the calculated descriptors.
    """
    mol_descriptors = {}
    for col, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)  # Generate a molecule object from the SMILES string
        if mol is not None:
            # Calculate each descriptor's value and store it in the dictionary
            mol_descriptors[col] = {desc_name: getattr(Descriptors, desc_name)(mol) for desc_name in desc_list}
    return mol_descriptors

def calculate_row(idx, row, mol_descriptors, salt_descriptors, scalers, desc_list):
    """
    Calculate the weighted average descriptor values for a single row of data.
    
    Args:
        idx (int): Index of the row.
        row (pd.Series): A row from the DataFrame.
        mol_descriptors (dict): Descriptors for molecules.
        salt_descriptors (dict): Descriptors for salts.
        scalers (dict): Scalers for normalizing descriptor values.
        desc_list (list): List of descriptor names.
    
    Returns:
        tuple: A tuple containing the row index and a dictionary of calculated descriptor values.
    """
    row_results = {}
    for desc_name in desc_list:
        values = []  # Store descriptor values
        weights = []  # Store weights
        for col in mol_descriptors.keys():
            if col in row and row[col] > 0:
                raw_value = mol_descriptors[col][desc_name]  # Get the raw descriptor value
                scaled_value = scalers[desc_name].transform([[raw_value]])[0][0]  # Normalize the descriptor value
                values.append(scaled_value)
                weights.append(row[col])

        if values and weights:
            # Calculate the weighted average
            weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        else:
            weighted_mean = None

        for col in salt_descriptors.keys():
            if col in row and row[col] > 0:
                raw_value = salt_descriptors[col][desc_name]
                scaled_value = scalers[desc_name].transform([[raw_value]])[0][0]
                if weighted_mean is not None:
                    weighted_mean += scaled_value
                break

        row_results[f"{desc_name}_mean"] = weighted_mean
    return idx, row_results

def process_data(input_file, smiles_dict_1, smiles_dict_2):
    """
    Process the data by calculating descriptors and saving the results.
    
    Args:
        input_file (str): Path to the input CSV file.
        smiles_dict_1 (dict): Dictionary of SMILES strings for non-salt compounds.
        smiles_dict_2 (dict): Dictionary of SMILES strings for salts.
    """
    data = pd.read_csv(input_file)  # Load the input file

    # Get all available descriptors and filter out unwanted ones
    all_desc_list = [x[0] for x in Descriptors._descList]
    excluded_desc = [
        "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
        "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"
    ]
    desc_list = [desc for desc in all_desc_list if desc not in excluded_desc]

    # Initialize the descriptor DataFrame
    descriptors_df = pd.DataFrame(index=data.index, columns=[f"{desc}_mean" for desc in desc_list])

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

    # Save the results to a new file
    output_file = input_file.replace('.csv', '_with_descriptors.csv')
    data_with_descriptors.to_csv(output_file, index=False)

def main():
    """
    Main program entry point.
    Get the input file path and SMILES dictionary file path from the command line arguments.
    If the number of command line arguments is incorrect, print usage instructions and exit the program.
    """
    if len(sys.argv) != 3:
        print("Usage: python RdKit_LASSO_S.py CALiSol-23_Dataset_cleaned_data.csv SMILES_dict.txt")
        sys.exit(1)

    input_file = sys.argv[1]  # Input file path
    smiles_dict_file = sys.argv[2]  # SMILES dictionary file path

    # Load SMILES dictionaries
    smiles_dict = load_smiles_dict(smiles_dict_file)
    smiles_dict_1 = {k: v for k, v in smiles_dict.items() if not k.startswith('Li')}  # Non-salt compounds
    smiles_dict_2 = {k: v for k, v in smiles_dict.items() if k.startswith('Li')}  # Salts

    # Process the data
    process_data(input_file, smiles_dict_1, smiles_dict_2)

if __name__ == "__main__":
    main()