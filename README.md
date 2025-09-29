# Electrolyte Ionic Conductivity Prediction

This repository contains a suite of Python scripts designed to process, analyze, and predict the ionic conductivity of electrolyte solutions using machine learning models. The scripts are organized to perform data cleaning, feature extraction, model training, and visualization.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
- [Running the Scripts](#running-the-scripts)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The scripts in this repository perform the following tasks:
1. **Data Cleaning and Grouping**: Divide the dataset into groups based on specific criteria and clean the data.
2. **Feature Extraction**: Calculate molecular descriptors using RDKit.
3. **LASSO Feature Selection**: Select important features using LASSO regression and visualize their importance.
4. **Model Training**: Train neural network models to predict ionic conductivity.
5. **Model Evaluation**: Evaluate the model's performance and visualize results.
6. **Random Search for Solvent Ratios**: Perform a random search to find the best solvent ratios for maximizing ionic conductivity.


## Dependencies

To run the scripts, you need the following libraries and tools:
- Python 3.8 or higher
- Pandas
- NumPy
- RDKit
- PyTorch
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Pickle

You can install the required libraries using pip:
```bash
pip install pandas numpy rdkit pytorch scikit-learn matplotlib seaborn shap
```

## Directory Structure

The repository is organized as follows:
```
electrolyte_conductivity/
├── CALiSol-23_Dataset.csv
├── SMILES_dict.txt
├── lasso_features_P.txt
├── divide_groups.py
├── clean_and_analyze_data.py
├── RdKit_LP.py
├── find_bestmodel_LP.py
├── LP.py
├── combined_script_LP.py
├── random_search_solvents.py
├── RdKit_LASSO_S.py
├── LASSO.py
├── LASSO_features_order.py
├── RdKit_LASSO_D.py
└── README.md
```

## Running the Scripts

The scripts should be run in the following order:

1. **Divide Data into Groups**:
   ```bash
   python divide_groups.py CALiSol-23_Dataset.csv
   ```
   This script groups the data based on specific criteria and saves each group as a separate CSV file.

2. **Clean and Analyze Data**:
   ```bash
   python clean_and_analyze_data.py CALiSol-23_Dataset.csv
   ```
   This script cleans the data, removes duplicates, and converts solvent ratios to mole fractions.

3. **Calculate Descriptors**:
   ```bash
   python RdKit_LP.py CALiSol-23_Dataset_cleaned_data.csv SMILES_dict.txt
   ```
   This script calculates molecular descriptors using RDKit and adds them to the dataset.

4. **Find the Best Model**:
   ```bash
   python find_bestmodel_LP.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv lasso_features_P.txt
   ```
   This script trains multiple neural network models and identifies the best-performing one.

5. **Train and Evaluate the Model**:
   ```bash
   python LP.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv lasso_features_P.txt
   ```
   This script trains the final model and evaluates its performance.

6. **Run the Combined Script**:
   ```bash
   python combined_script_LP.py
   ```
   This script combines the data cleaning, feature extraction, and model prediction steps into a single workflow.

7. **Plot MSLF vs Ionic Conductivity**:
   ```bash
   python plot_mslf_total.py output/summary.csv
   ```
   This script generates a scatter plot of MSLF vs ionic conductivity.

8. **Random Search for Solvent Ratios**:
   ```bash
   python random_search_solvents.py example_data.csv 1000
   ```
   This script performs a random search to find the best solvent ratios for maximizing ionic conductivity.

9. **LASSO Feature Selection (Single)**:
   ```bash
   python RdKit_LASSO_S.py CALiSol-23_Dataset_cleaned_data.csv SMILES_dict.txt
   python LASSO.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv SMILES_dict.txt
   python LASSO_features_order.py lasso_features.txt SMILES_dict.txt
   ```
   These scripts perform LASSO feature selection and visualize the importance of selected features. Through LASSO dimensionality reduction and Pearson correlation analysis for feature extraction, we can obtain lasso_features_P.txt

10. **LASSO Feature Selection (Double)**:
    ```bash
    python RdKit_LASSO_D.py CALiSol-23_Dataset_cleaned_data.csv SMILES_dict.txt
    python LASSO.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv SMILES_dict.txt
    python LASSO_features_order.py lasso_features.txt SMILES_dict.txt
    ```
    These scripts perform LASSO feature selection with double descriptors and visualize the importance of selected features.

## Usage

To use this repository, follow these steps:
1. Clone the repository.
2. Install the dependencies.
3. Run the scripts in the specified order.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.


