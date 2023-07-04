# Imports
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

# Transformers Imports
from transformers import AutoTokenizer



# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='dataset', help="Database directory for the concepts captions.")
args = parser.parse_args()

# Get the the dataset directory
base_dir = args.base_dir


# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=False)


# Go through "train" and "validation" splits and generate statistics
for split in ["train", "valid"]:
    captions_files = os.path.join(base_dir, f'ImageCLEFmedical_Caption_2023_caption_prediction_{split}_labels.csv')
    df = pd.read_csv(captions_files, sep='\t')

    token_lengths = []
    print(f"{split}...")
   
    print(f"Number of images/captions: {len(df)}")

    for caption in tqdm(df['caption'].values):
        caption_tokenized = tokenizer(caption)['input_ids']
        token_lengths.append(len(caption_tokenized))


    # Print statistics
    print(f"Avg/min/max sentence length (in tokens) is {np.mean(token_lengths):.1f}, {np.min(token_lengths):.1f}, {np.max(token_lengths):.1f}.")
    print(f"Median sentence length (in tokens) is {np.percentile(token_lengths, 50):.1f}.")
    print(f"First quartile is {np.percentile(token_lengths, 25):.1f}.")
    print(f"Last quartile is {np.percentile(token_lengths, 75):.1f}.")
    print(f"90th percentile is {np.percentile(token_lengths, 90):.1f}.")
    print(f"95th percentile is {np.percentile(token_lengths, 95):.1f}.")
    print(f"98th percentile is {np.percentile(token_lengths, 98):.1f}.")
    print(f"99th percentile is {np.percentile(token_lengths, 99):.1f}.")
    print()
