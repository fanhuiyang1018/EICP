#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   LASSO.py
@Time    :   2025/04/25 16:00:00
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
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
        if not content.startswith("{"):
            content = "{" + content
        if not content.endswith("}"):
            content = content + "}"
        try:
            smiles_dict = eval(content)
        except Exception as e:
            raise ValueError(f"Error parsing SMILES dictionary file: {e}")
    return smiles_dict

def load_and_clean_data(input_file, columns_to_exclude):
    """
    Load and clean the data.
    
    Args:
        input_file (str): Path to the input CSV file.
        columns_to_exclude (list): List of column names to exclude.
    
    Returns:
        pd.DataFrame: Cleaned data.
    """
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)

    data = data.drop(columns=[col for col in columns_to_exclude if col in data.columns], errors='ignore')
    return data

def standardize_data(X):
    """
    Standardize the data.
    
    Args:
        X (pd.DataFrame): Data to be standardized.
    
    Returns:
        np.ndarray: Standardized data.
        StandardScaler: Scaler object used for standardization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def select_important_features(X_scaled, y, alpha=0.01):
    """
    Select important features using Lasso regression.
    
    Args:
        X_scaled (np.ndarray): Standardized feature data.
        y (pd.Series): Target variable.
        alpha (float): Regularization parameter for Lasso.
    
    Returns:
        np.ndarray: Indices of important features.
        np.ndarray: Lasso regression coefficients.
    """
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    coefficients = lasso.coef_
    important_features_indices = np.argsort(np.abs(coefficients))[::-1]
    return important_features_indices, coefficients

def save_features_to_file(features, correlation_matrix, filename='lasso_features.txt'):
    """
    Save features and their correlations to a file.
    
    Args:
        features (list): List of feature names.
        correlation_matrix (pd.Series): Correlation values of features with the target variable.
        filename (str): Output file name.
    """
    with open(filename, 'w') as f:
        for feature in features:
            correlation = correlation_matrix.get(feature, np.nan)
            correlation = round(correlation, 3)
            f.write(f"{feature}: {correlation}\n")

def save_correlation_to_csv(correlation_matrix, output_csv):
    """
    Save the correlation matrix to a CSV file.
    
    Args:
        correlation_matrix (pd.Series): Correlation values of features with the target variable.
        output_csv (str): Output CSV file path.
    """
    if isinstance(correlation_matrix, pd.Series):
        correlation_df = correlation_matrix.reset_index()
        correlation_df.columns = ['Feature', 'Correlation']
    else:
        print("Error: correlation_matrix is not a Pandas Series.")
        return

    correlation_df.to_csv(output_csv, index=False)

def plot_correlation(correlation_matrix, important_features, output_image):
    """
    Plot the correlation of features with the target variable.
    
    Args:
        correlation_matrix (pd.Series): Correlation values of features with the target variable.
        important_features (list): List of important feature names.
        output_image (str): Output image file path.
    """
    correlation_values = correlation_matrix[important_features]
    sorted_features = correlation_values.sort_values(ascending=False).index
    correlation_values = correlation_values[sorted_features]

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("coolwarm", len(correlation_values))[::-1]
    
    sns.barplot(
        data=correlation_values.reset_index(), 
        x='index', 
        y=0, 
        hue='index', 
        palette=palette, 
        legend=False
    )
    
    plt.ylabel('Correlation Coefficient', fontsize=24, fontname='Times New Roman', fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45, fontsize=20, fontname='Times New Roman', fontweight='bold', ha='right')
    plt.yticks(fontsize=24, fontname='Times New Roman')
    
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()

def plot_pearson_correlation_heatmap(selected_features, data, output_image):
    """
    Plot the Pearson correlation coefficient heatmap of selected features and the target variable.
    
    Args:
        selected_features (list): List of selected feature names.
        data (pd.DataFrame): Original data.
        output_image (str): Output image file path.
    """
    correlation_matrix = data.corr()
    correlation_values = correlation_matrix.loc[selected_features, 'k']
    sorted_features = correlation_values.sort_values(ascending=False).index
    selected_data = data[sorted_features.tolist() + ['k']]

    corr_matrix = selected_data.corr()

    plt.figure(figsize=(20, 20))
    sns.set(font='Times New Roman', font_scale=1.2)
    sns.heatmap(
        corr_matrix, 
        annot=False, 
        fmt=".2f", 
        cmap='coolwarm', 
        square=True, 
        cbar_kws={"shrink": .5},
        annot_kws={"size": 20}
    )
    
    plt.xticks(fontsize=20, fontname='Times New Roman', rotation=45, ha='right')
    plt.yticks(fontsize=20, fontname='Times New Roman')
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()

def process_data(input_file, smiles_dict_file):
    """
    Process the data, select important features, and save the results.
    
    Args:
        input_file (str): Path to the input CSV file.
        smiles_dict_file (str): Path to the file containing the SMILES dictionary.
    """
    smiles_dict = load_smiles_dict(smiles_dict_file)
    columns_to_exclude = list(smiles_dict.keys())

    data = load_and_clean_data(input_file, columns_to_exclude)
    y = data['k']  # Target variable
    X = data.drop(columns=['k'])  # Feature data

    X_scaled, scaler = standardize_data(X)

    std_dev = np.std(X_scaled, axis=0)
    zero_std_features = X.columns[std_dev == 0]
    if len(zero_std_features) > 0:
        X = X.drop(columns=zero_std_features)
        X_scaled, scaler = standardize_data(X)

    important_features_indices, coefficients = select_important_features(X_scaled, y)
    important_features = X.columns[important_features_indices]

    correlation_matrix = X.corrwith(y)
    correlation_matrix = correlation_matrix.dropna()

    positive_correlation = correlation_matrix[correlation_matrix > 0].sort_values(ascending=False)
    negative_correlation = correlation_matrix[correlation_matrix < 0].sort_values(ascending=True)

    top_positive_features = positive_correlation.head(10).index
    top_negative_features = negative_correlation.head(10).index

    if 'T' in top_positive_features:
        top_positive_features = positive_correlation.head(11).index
    if 'c' in top_positive_features:
        top_positive_features = positive_correlation.head(11).index

    if 'T' in top_negative_features:
        top_negative_features = negative_correlation.head(11).index
    if 'c' in top_negative_features:
        top_negative_features = negative_correlation.head(11).index

    selected_features = list(top_positive_features) + list(top_negative_features)

    if 'c' not in selected_features:
        selected_features.append('c')
    if 'T' not in selected_features:
        selected_features.append('T')

    save_features_to_file(selected_features, correlation_matrix, 'lasso_features.txt')

    output_image = input_file.replace(".csv", "_correlation_plot.png")
    plot_correlation(correlation_matrix, selected_features, output_image)

    output_image_heatmap = input_file.replace(".csv", "_pearson_correlation_heatmap.png")
    plot_pearson_correlation_heatmap(selected_features, data, output_image_heatmap)

    output_csv = input_file.replace(".csv", "_correlation_data.csv")
    save_correlation_to_csv(correlation_matrix, output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python LASSO.py CALiSol-23_Dataset_cleaned_data_with_descriptors.csv SMILES_dict.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    smiles_dict_file = sys.argv[2]

    process_data(input_file, smiles_dict_file)
