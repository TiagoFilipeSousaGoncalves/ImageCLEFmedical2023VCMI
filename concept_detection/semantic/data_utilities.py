# Imports
import os
import numpy as np
import pandas as pd
import PIL

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Function: Map concepts to semantic types
def map_concepts_to_semantic(concept_detection_df, cui_semantic_types_df, column="cuis"):

    # Join the two concepts on "concept"
    new_df = concept_detection_df.copy().merge(right=cui_semantic_types_df.copy(), on=column)

    # Drop NaNs
    new_df = new_df.copy().dropna(axis=0)

    return new_df



# Function: Create subset according to semantic types
def get_semantic_concept_dataset(cui_semantic_types_csv, semantic_type, train_concept_detection_csv, val_concept_detection_csv, test_imgs_dir=None, subset='train'):

    assert ((test_imgs_dir is None) and (subset != 'test')) or ((test_imgs_dir is not None) and (subset=='test')), 'If you provide a directory for test images, you should select subset as test.'

    assert semantic_type in (
        'Body Part, Organ, or Organ Component', 
        'Disease or Syndrome', 
        'Diagnostic Procedure', 
        'Body Location or Region', 
        'Pathologic Function', 
        'Body Space or Junction', 
        'Neoplastic Process', 
        'Congenital Abnormality', 
        'Anatomical Abnormality', 
        'Medical Device', 
        'Functional Concept', 
        'Tissue', 
        'Body Substance', 
        'Acquired Abnormality', 
        'Qualitative Concept',
        'Body System',
        'Organ or Tissue Function',
        'Organism Function',
        'Manufactured Object',
        'Spatial Concept',
        'Substance'
    ), f"{semantic_type} not valid. Provide a valid semantic type."


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

    # Load the CSV with the CUIs Semantic Types
    cui_semantic_types_df = pd.read_csv(cui_semantic_types_csv, sep='\t')

    # Convert this into semantic types
    concept_detection_sem_df = map_concepts_to_semantic(
        concept_detection_df=concept_detection_df,
        cui_semantic_types_df=cui_semantic_types_df,
        column='cuis'
    )


    # Get the concepts that are related with the semantic type we want
    sem_type_concepts = list()
    for _, row in concept_detection_sem_df.iterrows():

        # Get concept_name
        concept_name = row["semantic_types"]

        # Check if this concept matches the semantic type
        if concept_name in (
        'Body Part, Organ, or Organ Component', 
        'Disease or Syndrome', 
        'Diagnostic Procedure', 
        'Body Location or Region', 
        'Pathologic Function', 
        'Body Space or Junction', 
        'Neoplastic Process', 
        'Congenital Abnormality', 
        'Anatomical Abnormality', 
        'Medical Device', 
        'Functional Concept', 
        'Tissue', 
        'Body Substance', 
        'Acquired Abnormality', 
        'Qualitative Concept',
        'Body System',
        'Organ or Tissue Function',
        'Organism Function',
        'Manufactured Object',
        'Spatial Concept',
        'Substance'
    ):
            if concept_name == semantic_type:
                sem_type_concepts.append(row["cuis"])

        elif semantic_type == "Miscellaneous Concepts":
            sem_type_concepts.append(row["cuis"])


    # Clean this list from duplicated values
    sem_type_concepts, _ = np.unique(ar=np.array(sem_type_concepts), return_counts=True)
    sem_type_concepts = list(sem_type_concepts)
    sem_type_concepts.sort()
    sem_type_concepts_dict = dict()
    inv_sem_type_concepts_dict = dict()

    # Create a dict for concept-mapping into classes
    for index, c in enumerate(sem_type_concepts):
        sem_type_concepts_dict[c] = index
        inv_sem_type_concepts_dict[index] = c


    # Get the formatted subset
    img_ids = list()
    img_labels = list()

    # Get the right subset
    if subset.lower() == 'train':
        subset_concept_detection_sem_df = train_concept_detection_df.copy()
    
    elif subset.lower() == 'validation':
        subset_concept_detection_sem_df = val_concept_detection_df.copy()
    
    else:
        subset_concept_detection_sem_df = test_concept_detection_df.copy()



    for index, row in subset_concept_detection_sem_df.iterrows():

        # Get image ids
        img_ids.append(row["ID"])

        # Get cuis
        if subset.lower() == 'test':
            img_labels = list()
        
        else:
            cuis = row["cuis"]
            cuis = cuis.split(';')

            # Create temporary concepts list to clean subset
            tmp_concepts = list()

            # Split the cuis
            for c in cuis:
                if c in sem_type_concepts:
                    tmp_concepts.append(c)

            tmp_concepts_unique, _ = np.unique(ar=np.array(tmp_concepts), return_counts=True)
            tmp_concepts_unique = list(tmp_concepts_unique)

            if len(tmp_concepts_unique) > 0:
                label = [sem_type_concepts_dict.get(i) for i in tmp_concepts_unique]
                img_labels.append(label)

            else:
                label = []
                img_labels.append(label)


    # Test Set
    if subset.lower() == 'test':
        img_labels = list()


    return img_ids, img_labels, sem_type_concepts_dict, inv_sem_type_concepts_dict



# Class: ImgClefConc Dataset
class ImgClefConcDataset(Dataset):
    def __init__(self, img_datapath, cui_semantic_types_csv, semantic_type, train_concept_detection_csv, val_concept_detection_csv, test_imgs_dir=None, transform=None, subset='train', classweights=None):

        # Get the desired dataset
        self.img_ids, self.img_labels, self.sem_type_concepts_dict, self.inv_sem_type_concepts_dict = get_semantic_concept_dataset(
            cui_semantic_types_csv=cui_semantic_types_csv,
            semantic_type=semantic_type,
            train_concept_detection_csv=train_concept_detection_csv,
            val_concept_detection_csv=val_concept_detection_csv,
            test_imgs_dir=test_imgs_dir,
            subset=subset
        )


        # Save into class variables
        self.subset = subset
        self.img_datapath = img_datapath
        self.nr_classes = len(self.sem_type_concepts_dict)
        self.transform = transform


        # Since we are dealing with a multilabel case
        if len(self.img_labels) > 0:
            matrix_labels = np.zeros((len(self.img_ids), len(self.sem_type_concepts_dict)))
            for r in range(len(self.img_ids)):
                label = self.img_labels[r]

                for c in label:
                    matrix_labels[r, c] = 1

            self.img_labels = matrix_labels.copy()


            # Compute class weights for loss function
            if classweights:
                
                # pos_weights = neg / pos
                pos_count = np.count_nonzero(matrix_labels.copy(), axis=0)
                neg_count = len(matrix_labels) - pos_count

                np.testing.assert_array_equal(np.ones_like(pos_count) * len(matrix_labels), np.sum((neg_count, pos_count), axis=0))
                pos_weights = neg_count / pos_count

                self.pos_weights = pos_weights
            
            else:
                self.pos_weights = None


        return


    # Method: __len__
    def __len__(self):
        return len(self.img_ids)

    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get images
        img_name = self.img_ids[idx]
        
        if self.subset in ('train', 'validation'):
            image = PIL.Image.open(os.path.join(self.img_datapath, f"{img_name}.jpg")).convert("RGB")
        else:
            image = PIL.Image.open(os.path.join(self.img_datapath, f"{img_name}")).convert("RGB")

        # Get labels
        if len(self.img_labels) > 0:
            label = self.img_labels[idx]
        
        else:
            label = list()
            label = torch.Tensor(label)

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label, img_name



# Run this script to test code
if __name__ == "__main__":

    # Get CUIs Semantic Types
    cui_semantic_types_csv = os.path.join('data', 'processed', "ImageCLEFmedical_Caption_2023_cui_semantic_types.csv")

    # Get train set 
    train_concept_detection_csv = os.path.join('data', 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv')

    # Get validation set
    val_concept_detection_csv = os.path.join('data', 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv')


    print("Train set:")
    img_ids, img_labels, sem_type_concepts_dict, inv_sem_type_concepts_dict = get_semantic_concept_dataset(
        cui_semantic_types_csv=cui_semantic_types_csv,
        semantic_type='Body Part, Organ, or Organ Component',
        train_concept_detection_csv=train_concept_detection_csv,
        val_concept_detection_csv=val_concept_detection_csv,
        subset='train'
    )
    print(len(img_ids), len(img_labels), len(sem_type_concepts_dict))

    print("Validation set:")
    img_ids, img_labels, sem_type_concepts_dict, inv_sem_type_concepts_dict = get_semantic_concept_dataset(
        cui_semantic_types_csv=cui_semantic_types_csv,
        semantic_type='Body Part, Organ, or Organ Component',
        train_concept_detection_csv=train_concept_detection_csv,
        val_concept_detection_csv=val_concept_detection_csv,
        subset='validation'
    )
    print(len(img_ids), len(img_labels), len(sem_type_concepts_dict))
