# Source: https://documentation.uts.nlm.nih.gov/scripts/get-concepts-for-a-list-of-codes.py
# Source: https://documentation.uts.nlm.nih.gov/rest/concept/
# This script returns UMLS CUIs based on an input file of codes, where each line in txt file is a separate code.
# All codes must be from the same vocabulary.
# If no results are found for a specific code, this will be noted in output and output file.
# Each set of results for a code is separated in the output file with '***'.



# Imports
import os
import argparse
import requests
import pandas as pd
from tqdm import tqdm



# CLIR
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='dataset', help="Database directory for the concepts labels.")
parser.add_argument("--processed_dir", type=str, default='dataset/processed/', help="Directory for the processed files (obtained from the original database).")
parser.add_argument("--api_key", type=str, help="API Key from your UTS Profile.")
args = parser.parse_args()



# Get dataset directories
base_dir = args.base_dir
processed_dir = args.processed_dir

# Create processed_dir if needed
if not os.path.isdir(processed_dir):
    os.makedirs(processed_dir)



# API Key: API Key from UTS Profile
API_KEY = args.api_key

# API Version: API Version
API_VERSION = 'current'

# Path of the input file
INPUT_FILE = os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_cui_mapping.csv')

# Path of the output file
OUTPUT_FILE = os.path.join(processed_dir, 'ImageCLEFmedical_Caption_2023_cui_semantic_types.csv')


# Base URI for the API
base_uri = 'https://uts-ws.nlm.nih.gov/rest'


# Create a list of the concept codes and names
cuis_df = pd.read_csv(INPUT_FILE, sep='\t', names=['cuis', 'names'])
cuis = list(cuis_df['cuis'].values)
names = list(cuis_df['names'].values)



# Generate the output file with the concepts related to the codes
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    
    # Create a file header
    f.write("cuis\tnames\tsemantic_types\n")

    for cui, name in tqdm(zip(cuis, names)):

        path = f'/content/{API_VERSION}/CUI/{cui}'
        query = {'apiKey':API_KEY}
        output = requests.get(base_uri+path, params=query)
        output.encoding = 'utf-8'

        # Get output in JSON format
        outputJson = output.json()
        
        # Convert output in dictionary format
        results = outputJson['result']

        
        # In case we don't find any result
        if len(results) == 0:
            f.write(f'{cui}\tN/A\tN/A')


        # In case we find results
        else:
            semantic_type = results['semanticTypes'][0]['name']
            f.write(f'{cui}\t{name}\t{semantic_type}')
            f.write('\n')



print('Finished')
