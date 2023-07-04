# Imports
import os
import pandas as pd



# Function: Read concepts CSV
def read_concepts_csv(csv_path):

    # Open .CSV path of the concepts
    concepts_df = pd.read_csv(filepath_or_buffer=csv_path, sep='\t')

    return concepts_df



if __name__ == "__main__":

    # Imports
    import argparse



    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default='dataset', help="Database directory for the concepts labels.")
    parser.add_argument("--processed_dir", type=str, default='dataset/processed/', help="Directory for the processed files (obtained from the original database).")
    args = parser.parse_args()

    # Get database directories
    base_dir = args.base_dir
    processed_dir = args.processed_dir

    # Create processed_dir if needed
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)


    # Get the paths of the .CSV files
    train_csv = os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv')
    val_csv = os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv')


    # Read .CSV files
    train_df = read_concepts_csv(csv_path=train_csv)
    val_df = read_concepts_csv(csv_path=val_csv)
    
    # Get CUIs
    train_concepts = train_df['cuis'].values
    val_concepts = val_df['cuis'].values

    # Concatenate these lists
    raw_concepts = list(train_concepts) + list(val_concepts)
    proc_concepts = list()

    # Parse concepts
    for c in raw_concepts:
        proc_concepts += c.split(';')
    
    # Remove white spaces (if there are any)
    proc_concepts = [c.strip() for c in proc_concepts]


    # Get unique concepts (remove duplicates)
    un_concepts = list(dict.fromkeys(proc_concepts))
    un_concepts.sort()

    # Open file to save this
    with open(os.path.join(processed_dir, "unique_concepts.csv"), "w") as f:
        f.write('cuis')
        f.write('\n')
        for concept in un_concepts:
            f.write(concept)
            f.write('\n')
