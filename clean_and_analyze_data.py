#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   clean_and_analyze_data.py
@Time    :   2025/04/25 15:40:49
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib parameters for better visualization
rc = plt.rcParams
rc['font.family'] = 'Times New Roman'
size = 26
rc['font.size'] = size
rc['axes.labelsize'] = size
rc['axes.titlesize'] = size
rc['axes.labelweight'] = 'normal'
rc['xtick.labelsize'] = size
rc['ytick.labelsize'] = size
rc['xtick.direction'] = 'in'
rc['ytick.direction'] = 'in'
rc['axes.linewidth'] = 2

light_blue = '#0072C6'

def clean_data(input_file):
    """
    Main function to clean the data from the input CSV file.
    It performs the following operations:
    1. Remove missing values and duplicates.
    2. Filter out rows that do not meet certain conditions.
    3. Convert data types.
    4. Convert solvent ratios to mole fractions.
    5. Convert salt concentration units.
    6. One-hot encode the salt type.
    7. Drop unnecessary columns.
    8. Save the cleaned data to a new file.
    """
    # Read the input CSV file
    data = pd.read_csv(input_file)

    # Remove missing values
    data = data.dropna()

    # Remove duplicates
    data = data.drop_duplicates()

    # Filter out rows that do not meet the conditions
    # Keep rows where the 'k' column is greater than or equal to 1e-5
    data = data[data['k'] >= 1e-5]
    # Keep rows where the 'T', 'c', and 'k' columns are greater than 0
    data = data[(data['T'] > 0) & (data['c'] > 0) & (data['k'] > 0)]

    # Strip leading and trailing whitespace from the 'salt' column
    data['salt'] = data['salt'].str.strip()

    # Convert data types
    data['T'] = data['T'].astype(float)  # Convert temperature to float
    data['c'] = data['c'].astype(float)  # Convert concentration to float
    data['salt'] = data['salt'].astype(str)  # Convert salt type to string
    data['c units'] = data['c units'].astype(str)  # Convert concentration units to string
    data['solvent ratio type'] = data['solvent ratio type'].astype(str)  # Convert solvent ratio type to string

    # Define solvents with their molar masses (g/mol) and densities (g/mL)
    solvents = [
        ('EC', 88.06, 1.3210), ('PC', 102.08, 1.205), ('DMC', 90.08, 1.07),
        ('EMC', 104.10, 0.902), ('DEC', 118.132, 0.975), ('DME', 90.12, 0.86),
        ('DMSO', 78.13, 1.1004), ('AN', 41.05, 0.786), ('MOEMC', 134.13, 1.5),
        ('TTFP', 344.07, 1.487), ('EA', 88.10, 0.902), ('MA', 74.08, 0.932),
        ('FEC', 106.05, 1.454), ('DOL', 74.08, 1.06), ('2-MeTHF', 86.13, 0.854),
        ('DMM', 162.2, 0.902), ('Freon 11', 137.36, 1.49), ('Methylene chloride', 84.93, 1.33),
        ('THF', 72.10, 0.888), ('Toluene', 92.14, 0.867), ('Sulfolane', 120.17, 1.26),
        ('2-Glyme', 134.17, 0.937), ('3-Glyme', 178.22, 0.986), ('4-Glyme', 222.28, 1.009),
        ('3-Me-2-Oxazolidinone', 101.10, 1.17), ('3-MeSulfolane', 134.20, 1.20),
        ('Ethyldiglyme', 134.17, 0.937), ('DMF', 73.09, 0.944), ('Ethylbenzene', 106.17, 0.866),
        ('Ethylmonoglyme', 76.10, 0.965), ('Benzene', 78.11, 0.876), ('g-Butyrolactone', 86.09, 1.13),
        ('Cumene', 120.19, 0.862), ('Propylsulfone', 150.24, 1.109), ('Pseudocumeme', 120.19, 0.876),
        ('TEOS', 208.33, 0.940), ('m-Xylene', 106.17, 0.860), ('o-Xylene', 106.16, 0.87596)
    ]

    # Extract the list of solvent names
    solvent_columns = [solvent[0] for solvent in solvents]

    def convert_solvent_ratio(row):
        """
        Convert solvent ratios to mole fractions.
        Depending on the value in the 'solvent ratio type' column ('w', 'v', or 'mol'),
        calculate the moles of each solvent and normalize the total moles to 1.
        """
        total_moles = 0  # Total moles
        solvent_moles = {}  # Moles of each solvent

        for solvent in solvents:
            name, molar_weight, density = solvent
            if row[name] > 0:  # If the value of the solvent is greater than 0
                if row['solvent ratio type'] == 'w':  # Mass fraction
                    moles = row[name] / molar_weight
                elif row['solvent ratio type'] == 'v':  # Volume fraction
                    moles = (row[name] * density) / molar_weight
                elif row['solvent ratio type'] == 'mol':  # Mole fraction
                    moles = row[name]
                else:
                    raise ValueError(f"Unknown solvent ratio type: {row['solvent ratio type']}")

                solvent_moles[name] = moles
                total_moles += moles

        if total_moles == 0:  # If total moles is 0, return the original row
            return row

        # Normalize the moles of each solvent to 1
        for solvent in solvent_moles:
            row[solvent] = solvent_moles[solvent] / total_moles

        # Update the 'solvent ratio type' column to 'mol'
        row['solvent ratio type'] = 'mol'

        return row

    # Apply the convert_solvent_ratio function to convert solvent ratios
    data = data.apply(convert_solvent_ratio, axis=1)

    def convert_salt_concentration(row):
        """
        Convert salt concentration units from mol/kg to mol/L.
        Calculate the total mass of the mixed solvents based on their masses and densities,
        then calculate the density of the mixed solvents and convert the concentration to mol/L.
        """
        if row['c units'] == 'mol/kg':
            total_mass = 0  # Total mass of the mixed solvents

            # Calculate the mass of each solvent
            for solvent in solvent_columns:
                if row[solvent] > 0:
                    molar_weight = solvents[solvent_columns.index(solvent)][1]
                    density = solvents[solvent_columns.index(solvent)][2]
                    total_mass += row[solvent] * molar_weight * density

            if total_mass == 0:  # If total mass is 0, return the original row
                return row

            # Calculate the density of the mixed solvents
            mixed_density = total_mass

            # Convert concentration units
            row['c'] = row['c'] / (mixed_density / 1000)
            row['c units'] = 'mol/L'

        return row

    # Apply the convert_salt_concentration function to convert salt concentration units
    data = data.apply(convert_salt_concentration, axis=1)

    # One-hot encode the 'salt' column without prefix
    data = pd.get_dummies(data, columns=['salt'], prefix='')

    # Remove underscores from column names
    data.columns = [col.replace('_', '') for col in data.columns]

    # Drop unnecessary columns
    columns_to_drop = ['doi', 'c units', 'solvent ratio type']
    data = data.drop(columns=columns_to_drop)

    # Generate the output file name
    output_file = input_file.replace(".csv", "_cleaned_data.csv")

    # Save the cleaned data to a new file
    data.to_csv(output_file, index=False)

    return data

def plot_ionic_conductivity_histogram(data, output_dir):
    """
    Plot a histogram of ionic conductivity.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data['k'], bins=50, log_scale=True, color=light_blue)
    plt.xlabel('Ionic Conductivity (mS/cm)', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.gca().tick_params(axis='x', direction='out')
    
    # Save the figure to the specified directory
    figure_path = os.path.join(output_dir, 'Ionic_Conductivity_Histogram.png')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()

def plot_ionic_conductivity_vs_1000T_scatter(data, output_dir):
    """
    Plot a scatter plot of ionic conductivity vs 1000/T.
    """
    data['1000/T'] = 1000 / data['T']
    plt.figure(figsize=(8, 6))
    plt.scatter(data['1000/T'], data['k'], s=50, color=light_blue)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('1000/Temperature (K⁻¹)', fontweight='bold')
    plt.ylabel('Ionic Conductivity (mS/cm)', fontweight='bold')
    
    # Save the figure to the specified directory
    figure_path = os.path.join(output_dir, 'Ionic_Conductivity_vs_1000T_Scatter.png')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()

def main():
    """
    Main program entry point.
    Get the input file path from the command line arguments.
    If the number of command line arguments is incorrect, print usage instructions and exit the program.
    """
    if len(sys.argv) != 2:
        print("Usage: python clean_and_analyze_data.py CALiSol-23_Dataset.csv")
        sys.exit(1)

    # Get the input file path
    input_file = sys.argv[1]
    input_dir = os.path.dirname(input_file)  # Get the directory of the input file

    # Create a subdirectory for figures if it doesn't exist
    figures_dir = os.path.join(input_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Clean the data
    cleaned_data = clean_data(input_file)

    # Plot the cleaned data
    plot_ionic_conductivity_histogram(cleaned_data, figures_dir)
    plot_ionic_conductivity_vs_1000T_scatter(cleaned_data, figures_dir)

if __name__ == "__main__":
    main()