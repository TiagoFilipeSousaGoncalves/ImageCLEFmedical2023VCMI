# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Function: Get top-k concepts in dataset
def get_top_k_concepts(concept_detection_df, k=100):

    # Get all concepts in the database
    all_concepts = concept_detection_df["cuis"].values

    # Create a dictionary of concept frequencies
    concept_freqs = dict()

    # Iterate through all concepts to create this a list of unique concepts
    for r_concepts in all_concepts:
        
        # Get concepts
        parsed_r_concepts = r_concepts.split(';')
        
        for c in parsed_r_concepts:

            if c in concept_freqs.keys():
                concept_freqs[c] += 1
            else:
                concept_freqs[c] = 1
    

    # Get the top-k concepts
    sorted_concept_freqs = dict(sorted(concept_freqs.items(), key=lambda item: item[1], reverse=True))
    top_k_concepts = list(sorted_concept_freqs.keys())[0:100]


    return top_k_concepts



# Class: ImageConceptDataset
class ImageConceptDataset(Dataset):
    def __init__(self, images_dir, train_concept_detection_csv, val_concept_detection_csv, subset, test_imgs_dir=None, top_k=None, transform=None):


        assert subset in ('train', 'valid', 'test', 'complete'), 'Please provide a valid subset.'
        

        # Load the the CSVs with the concepts
        train_concept_detection_df = pd.read_csv(train_concept_detection_csv, sep='\t')
        val_concept_detection_df = pd.read_csv(val_concept_detection_csv, sep='\t')

        # If subset is test
        if subset == 'test':
            test_imgs = [i for i in os.listdir(test_imgs_dir) if not i.startswith('.')]
            test_concept_detection_dict = {'ID':test_imgs}
            test_concept_detection_df = pd.DataFrame.from_dict(test_concept_detection_dict)
        

        # Create a full concept dataframe for the dictionaries and the training of the models
        concept_detection_df = pd.concat([train_concept_detection_df, val_concept_detection_df], ignore_index=True)


        # Then, we create a dictionary to store a mapping between the class-index and the concept
        all_concepts_dict = dict()
        all_concepts_inv_dict = dict()
        all_unique_concepts_list = list()


        # Get all the concepts available in our dataset (train + validation)
        if top_k is not None:
            top_k_concepts = get_top_k_concepts(concept_detection_df=concept_detection_df, k=top_k)


        # Get all concepts
        all_concepts = concept_detection_df["cuis"].values


        # Iterate through all concepts to create this a list of unique concepts
        for r_concepts in all_concepts:
            
            # Get concepts
            parsed_r_concepts = r_concepts.split(';')
            for c in parsed_r_concepts:
                if top_k is not None:
                    if c in top_k_concepts:
                        all_unique_concepts_list.append(c)
                else:
                    all_unique_concepts_list.append(c)


        # Clean list and get only unique values
        all_unique_concepts_list = list(dict.fromkeys(all_unique_concepts_list))

        # Create dictionaries
        for c_idx, c in enumerate(all_unique_concepts_list):
            all_concepts_dict[c_idx] = c
            all_concepts_inv_dict[c] = c_idx


        # Generate matrix w/ labels
        if subset in ('train', 'valid', 'complete'):

            if subset == 'train':
                subset_df = train_concept_detection_df.copy()
            elif subset == 'valid':
                subset_df = val_concept_detection_df.copy()
            else:
                subset_df = concept_detection_df.copy()


            # Create a matrix of labels
            subset_concepts = subset_df["cuis"].values.copy()
            matrix = np.zeros((len(subset_concepts), len(all_unique_concepts_list)))
            for i in range(len(subset_concepts)):
                image_concepts = subset_concepts[i].split(';')
                for c in image_concepts:
                    if c in all_unique_concepts_list:
                        matrix[i][all_concepts_inv_dict[c]] = 1
            

            # Assign to the labels variable
            self.labels = matrix.copy()

        else:
            subset_df = test_concept_detection_df.copy()
            self.labels = list()
        

        # Assign the remaining variables
        self.image_names = list(subset_df["ID"].values)
        self.subset = subset
        self.subset_images_dir = os.path.join(images_dir, subset) if subset in ('train', 'valid', 'test') else images_dir
        self.nr_classes = len(all_concepts_dict.keys())
        self.all_concepts_dict = all_concepts_dict
        self.all_concepts_inv_dict = all_concepts_inv_dict
        self.transform = transform


    # Method: __len__
    def __len__(self):
        return len(self.image_names)


    # Method: __getitem__
    def __getitem__(self, idx):

        # Get image ID
        image_id = self.image_names[idx]

        # Load image
        if self.subset in ('train', 'valid'):
            image = Image.open(os.path.join(self.subset_images_dir, self.image_names[idx]+'.jpg')).convert("RGB")
        
        elif self.subset == 'complete':
            if os.path.exists(os.path.join(self.subset_images_dir, 'train', self.image_names[idx]+'.jpg')):
                image = Image.open(os.path.join(self.subset_images_dir, 'train', self.image_names[idx]+'.jpg')).convert("RGB")
            else:
                image = Image.open(os.path.join(self.subset_images_dir, 'valid', self.image_names[idx]+'.jpg')).convert("RGB")
        
        else:
            image = Image.open(os.path.join(self.subset_images_dir, self.image_names[idx])).convert("RGB")

        # Apply image transforms (if necessary)
        if self.transform:
            image = self.transform(image)


        if len(self.labels) > 0:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32)
        
        else:
            label = list()
            label = torch.tensor(label)
        

        return image, label, image_id



if __name__ == "__main__":

    # Imports
    import argparse

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default='dataset', help="Database directory for the concepts captions.")
    parser.add_argument("--images_dir", type=str, default='dataset/images', help="Database directory for the images.")
    args = parser.parse_args()


    # Directories
    base_dir = args.base_dir
    images_dir = args.images_dir


    train = ImageConceptDataset(
        images_dir = images_dir,
        train_concept_detection_csv=os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
        val_concept_detection_csv=os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
        subset='train'
    )

    validation = ImageConceptDataset(
        images_dir = images_dir,
        train_concept_detection_csv=os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
        val_concept_detection_csv=os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
        subset='valid'
    )

    test = ImageConceptDataset(
        images_dir = images_dir,
        train_concept_detection_csv=os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
        val_concept_detection_csv=os.path.join(base_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
        subset='test',
        test_imgs_dir=os.path.join(images_dir, 'test')
    )
