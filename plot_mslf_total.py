#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   plot_mslf_total.py
@Time    :   2025/04/25 16:00:00
@Author  :   Your Name
@Email   :   your_email@example.com
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Set matplotlib parameters for better visualization
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 28,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
})

def load_data(file_path):
    """
    Load data from the input CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def plot_mslf_vs_ionic_conductivity(data, figures_dir):
    """
    Plot scatter plot of MSLF vs Ionic Conductivity.
    """
    # Separate data for 'predict.csv'
    predict_data = data[data['input_file'] == 'predict.csv']
    data = data[data['input_file'] != 'predict.csv']  # Remove 'predict.csv' from the main data

    # Extract MSLF, experimental, and predicted values for all data
    mslf = data['MSLF']
    k_exp = data['k_value_exp']
    k_pred = data['k_value_pred']

    # Extract MSLF, experimental, and predicted values for 'predict.csv'
    mslf_predict = predict_data['MSLF']
    k_exp_predict = predict_data['k_value_exp']
    k_pred_predict = predict_data['k_value_pred']

    # Plot scatter plot
    plt.figure(figsize=(8, 6))

    # Experimental values for all data are represented by blue circles
    plt.scatter(mslf, k_exp, color='blue', marker='o', label='Experimental', s=100, linewidth=1, zorder=3)

    # Predicted values for all data are represented by red triangles
    plt.scatter(mslf, k_pred, color='red', marker='^', label='Predicted', s=100, linewidth=1, zorder=3)

    # Experimental values for 'predict.csv' are represented by green stars
    plt.scatter(mslf_predict, k_exp_predict, color='green', marker='*', label='E_This work', s=200, linewidth=1, zorder=4)

    # Predicted values for 'predict.csv' are represented by purple stars
    plt.scatter(mslf_predict, k_pred_predict, color='purple', marker='*', label='P_This work', s=200, linewidth=1, zorder=4)

    # Add titles and labels
    plt.xlabel('MSLF', fontweight='bold')
    plt.ylabel('Ionic Conductivity (mS/cm)', fontweight='bold')

    # Add legend
    plt.legend(loc='upper left', fontsize=20, frameon=False)

    # Adjust borders
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # Save the figure
    output_path = os.path.join(figures_dir, 'mslf_total.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Plot scatter plot of MSLF vs Ionic Conductivity")
    parser.add_argument("file_path", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    # Load data
    data = load_data(args.file_path)

    # Create the figures directory in the script's running directory
    figures_dir = os.path.join(os.getcwd(), 'figures')  # Use the current working directory
    os.makedirs(figures_dir, exist_ok=True)

    # Plot MSLF vs Ionic Conductivity
    plot_mslf_vs_ionic_conductivity(data, figures_dir)

if __name__ == "__main__":
    main()