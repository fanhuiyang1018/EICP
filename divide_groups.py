#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   divide_groups.py
@Time    :   2025/04/25 16:24:39
@Author  :   Huiyang Fan 
@Email   :   12227047@zju.edu.cn
"""

import pandas as pd
import sys
import os

# List of solvent names
solvents = [
    'EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC', 'TTFP', 'EA', 'MA',
    'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11', 'Methylene chloride', 'THF', 'Toluene', 
    'Sulfolane', '2-Glyme', '3-Glyme', '4-Glyme', '3-Me-2-Oxazolidinone', '3-MeSulfolane',
    'Ethyldiglyme', 'DMF', 'Ethylbenzene', 'Ethylmonoglyme', 'Benzene', 'g-Butyrolactone',
    'Cumene', 'Propylsulfone', 'Pseudocumeme', 'TEOS', 'm-Xylene', 'o-Xylene'
]

# Get the input file path from command line arguments
input_csv = sys.argv[1]
output_prefix = 'group'

# Ensure the output directory exists
output_dir = 'groups'
os.makedirs(output_dir, exist_ok=True)

# Read the input CSV file
df = pd.read_csv(input_csv)

# Initialize variables for grouping
group_index = 1
current_doi = None
current_salt = None
current_solvent_ratio_type = None
current_solvent_values = {solvent: None for solvent in solvents}
group_df = pd.DataFrame(columns=df.columns)

# Iterate through the DataFrame to group and save files
for index, row in df.iterrows():
    # Check if the current row meets the specified conditions
    if 296.15 <= row['T'] <= 300.15 and 0.9 <= row['c'] <= 1.1:
        # Check if any of the grouping criteria have changed
        if (row['doi'] != current_doi or 
            row['salt'] != current_salt or 
            row['solvent ratio type'] != current_solvent_ratio_type or 
            any(row[solvent] != current_solvent_values[solvent] for solvent in solvents)):
            
            if index > 0:
                # Save the current group DataFrame to a CSV file
                group_df.to_csv(os.path.join(output_dir, f'{output_prefix}{group_index}.csv'), index=False)
                group_index += 1
            
            # Update the current group information
            current_doi = row['doi']
            current_salt = row['salt']
            current_solvent_ratio_type = row['solvent ratio type']
            current_solvent_values = {solvent: row[solvent] for solvent in solvents}
            group_df = pd.DataFrame(columns=df.columns)
        
        # Add the current row to the group DataFrame
        group_df = pd.concat([group_df, row.to_frame().T], ignore_index=True)

# Save the last group DataFrame to a CSV file
if not group_df.empty:
    group_df.to_csv(os.path.join(output_dir, f'{output_prefix}{group_index}.csv'), index=False)

print(f'The grouping is complete. A total of {group_index} files have been generated in the "{output_dir}" directory.')