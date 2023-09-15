# Imports
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt



# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


    
# CLI Arguments
parser = argparse.ArgumentParser()

# CSV with CUIS and their Semantic Types
parser.add_argument('--cui_semantic_csv', type=str, default="ImageCLEFmedical_Caption_2023_cui_semantic_types.csv", help="Path to the CSV with CUIS and their Semantic Types.")

# Parse the arguments
args = parser.parse_args()
CUI_SEMANTIC_CSV = args.cui_semantic_csv


# Load CSV files
cui_semantic_csv_df = pd.read_csv(CUI_SEMANTIC_CSV, sep="\t", header=0)
# print(cui_semantic_csv_df.head())

# Get unique semantic types
semantic_types = cui_semantic_csv_df.copy()[['semantic_types']].values
semantic_types = semantic_types.flatten()
print('Length of the database: ', len(semantic_types))


# Get unique semantic types and their frequencies
semantic_types_dict = dict()

for st in semantic_types:
    if st not in semantic_types_dict.keys():
        semantic_types_dict[st] = 1
    else:
        semantic_types_dict[st] += 1


# Get this dictionary sorted by values
semantic_types_dict = dict(sorted(semantic_types_dict.items(), key=lambda item: item[1], reverse=True))
print('Number of unique values: ', len(semantic_types_dict))
print('Unique values: ', semantic_types_dict.keys())
print('Frequencies of the semantic types:', semantic_types_dict)


# Plot these results
sanity_check_sum = 0
bar_indices = list()
bar_values = list()
bar_labels = list()
idx = 0

for key, value in semantic_types_dict.items():
    
    # Sanity check
    sanity_check_sum += value

    # Save values in list
    bar_indices.append(idx)
    bar_values.append(value)
    bar_labels.append(key)

    # Update index
    idx += 1


print(f'Sanity check passes test: {sanity_check_sum==len(semantic_types)}')



# Plot the a bar plot
plt.bar(x=bar_indices, height=bar_values, label=bar_labels, tick_label=bar_labels)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig('results/data-analysis/semantic_types_freq.png', bbox_inches='tight')
plt.show()
