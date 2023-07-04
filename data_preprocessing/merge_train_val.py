# Imports
import os
import argparse
import pandas as pd



# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='dataset', help="Database directory for the concepts captions.")
parser.add_argument("--base_file", type=str, default='ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv', help="The training csv with the concept dectection labels.")
args = parser.parse_args()


# Get directory and files
BASE_DIR = args.base_dir
BASE_FILE = args.base_file



# Generate full database (train + validation)
train_df = pd.read_csv(os.path.join(BASE_DIR, BASE_FILE), sep="\t")
val_df = pd.read_csv(os.path.join(BASE_DIR,BASE_FILE.replace("train", "valid")), sep="\t")
new_df = pd.concat([train_df, val_df])
new_df = new_df.reset_index(drop=True)
new_df.to_csv(os.path.join(BASE_DIR, BASE_FILE.replace("train", "trainval")), index=False, sep="\t")
