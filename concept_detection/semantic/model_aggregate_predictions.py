# Imports
import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm



# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()



# Subset of the aggregated predictions
parser.add_argument('--subset', type=str, required=True, choices=['train', 'validation', 'test'], help="Predictions subset (train, validation or test).")

# Path to save the aggregated predictions
parser.add_argument('--aggregated_pred_dir', type=str, required=True, help="Directory to save aggregated predictions.")

# Paths of the .CSVs with the predictions
parser.add_argument('--pred_csvs_paths', action="append", type=str, required=True, help="Path to the .CSVs with predictions.")

# Parse the arguments
args = parser.parse_args()



# Subset
subset = args.subset

# Directory of the aggregated predictions
aggregated_pred_dir = args.aggregated_pred_dir

# CSV files
list_csv_paths = args.pred_csvs_paths
print("Number of models: ", len(list_csv_paths))

# Pre-create image lists
eval_data = dict()

# Go through all the .CSV files
print('Aggregating...')
for csv_path in tqdm(list_csv_paths):

    # Read .CSV
    df = pd.read_csv(os.path.join(csv_path, f"{subset}_preds.csv"), sep="|", header=None)

    # Create image list
    for _, row in df.iterrows():
        if row[0] not in eval_data.keys():
            eval_data[row[0]] = list()

    # Append concepts (if different from 'None')
    for index, row in df.iterrows():

        eval_data[row[0]] += str(row[1]).split(';')

        for i, c in enumerate(eval_data[row[0]]):
            if c in ("None", "nan"):
                eval_data[row[0]].pop(i)

        # Remove duplicates if needed (we don't know why this happens)
        eval_data[row[0]] = list(dict.fromkeys(eval_data[row[0]]))

# Process concept lists
for key, value in eval_data.items():
    # Add the valid concepts
    predicted_concepts = ""
    for c in value:
        predicted_concepts += f"{c};"

    eval_data[key] = predicted_concepts[:-1]

# Convert this data into a DataFrame
df_dict = dict()
df_dict["ID"] = list()
df_dict["cuis"] = list()
for key, value in eval_data.items():
    df_dict["ID"].append(key)
    df_dict["cuis"].append(value)

evaluation_df = pd.DataFrame(data=df_dict)
fname = os.path.join(aggregated_pred_dir, f"{subset}_preds_agg.csv")
evaluation_df.to_csv(fname, sep="|", index=False, header=False)
print(f'Saved results to: {fname}')
