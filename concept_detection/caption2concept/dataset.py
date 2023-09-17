# Imports
import os
import pandas as pd
from PIL import Image
import numpy as np

# PyTorch Imports
import torch



# Class: CustomDataCollator
class CustomDataCollator:

    # Method: __init__
    def __init__(self, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length


    # Method: __call__
    def __call__(self, batch):

        concepts = [i['labels'] for i in batch]
        concepts = torch.stack(concepts)

        captions = [i['input_ids'] for i in batch]
        captions = self.tokenizer(
            captions,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        out_batch = {
            'labels': concepts,
            'input_ids': captions['input_ids'],
            'attention_mask': captions['attention_mask']
        }

        return out_batch



# Class: Dataset
class Dataset(torch.utils.data.Dataset):
    
    # Method: __init__
    def __init__(
            self,
            gt_file,
            concepts_file,
            df_all_concepts=None
    ):
        super(Dataset, self).__init__()
        self.df = pd.read_csv(gt_file, sep='\t')
        self.df_concepts = pd.read_csv(concepts_file, sep='\t')
        self.df_all_concepts = pd.read_csv(df_all_concepts, sep=",")

        all_concepts = self.df_all_concepts["concept"]
        dict_concept = dict()
        for idx, c in enumerate(all_concepts):
            dict_concept[c] = idx

        matrix = np.zeros((len(self.df_concepts["ID"]), len(all_concepts)))
        for i in range(len(self.df_concepts["ID"])):
            dict_concepts_per_image = self.df_concepts["cuis"][i].split(";")
            for c in dict_concepts_per_image:
                matrix[i][dict_concept[c]] = 1

        self.labels = matrix


    # Method: __getitem__
    def __getitem__(self, index):
        df_row = self.df.iloc[index]

        sample = {}

        # concepts
        sample['labels'] = torch.tensor(self.labels[index], dtype=torch.float32)

        # text
        sample['input_ids'] = df_row['caption']

        return sample


    # Method: __len__
    def __len__(self):
        return len(self.df)



# Class: EvalDataset
class EvalDataset(torch.utils.data.Dataset):
    
    # Method: __init__
    def __init__(
            self,
            base_path,
            img_processor=None,
    ):
        super(EvalDataset, self).__init__()
        self.base_path = base_path
        self.imgs = [r for r in os.listdir(self.base_path) if r.endswith('.jpg') and not r.startswith('.')]

        self.img_processor = img_processor


    # Method: __getitem__
    def __getitem__(self, index):
        imgid = str(self.imgs[index])

        # Image
        img_path = os.path.join(
            self.base_path,
            imgid
        )

        img = Image.open(img_path).convert('RGB')

        if self.img_processor:
            img = self.img_processor(img, return_tensors="pt")

        pixel_values = img['pixel_values'].squeeze(0)

        return {'id': imgid[:-4], 'pixel_values': pixel_values}


    # Method: __len__
    def __len__(self):
        return len(self.imgs)
