# Imports
import os
import pandas as pd
import numpy as np

# PyTorch Imports
import torch
import torchxrayvision as xrv

# Skimage Imports
import skimage



# Class: CustomDataCollator
class CustomDataCollator:
    def __init__(self, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        imgs = [i["pixel_values"] for i in batch]
        imgs = torch.stack(imgs)

        captions = [self.tokenizer.bos_token + i["labels"] + self.tokenizer.eos_token for i in batch]
        captions = self.tokenizer(
            captions,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.where(captions.attention_mask == 0, -100, captions.input_ids) # ignore padding tokens in loss

        clf_labels = None
        if "clf_labels" in batch[0]:
            clf_labels = [i["clf_labels"] for i in batch]
            clf_labels = torch.stack(clf_labels)

        out_batch = {
            "pixel_values": imgs,
            "input_ids": captions["input_ids"],
            "attention_mask": captions["attention_mask"],
            "labels": labels,
            "clf_labels": clf_labels
        }

        return out_batch



# Class: Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        gt_file,
        concepts_file=None,
        df_all_concepts=None,
        transform=None,
    ):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.trainval = 'trainval' in gt_file
        self.df = pd.read_csv(gt_file, sep="\t")
        # self.df = self.df[:500] 

        if concepts_file and df_all_concepts:
            self.df_concepts = pd.read_csv(concepts_file, sep='\t')
            self.df_all_concepts = pd.read_csv(df_all_concepts)

            all_concepts = self.df_all_concepts["cuis"]
            dict_concept = dict()
            for idx, c in enumerate(all_concepts):
                dict_concept[c] = idx

            matrix = np.zeros((len(self.df_concepts["ID"]), len(all_concepts)))
            for i in range(len(self.df_concepts["ID"])):
                dict_concepts_per_image = self.df_concepts["cuis"][i].split(";")
                for c in dict_concepts_per_image:
                    matrix[i][dict_concept[c]] = 1

            self.clf_labels = matrix

        self.transform = transform

    def __getitem__(self, index):

        df_row = self.df.iloc[index]

        sample = {}

        imgid = str(df_row["ID"])

        # image
        if self.trainval:
            img_path = os.path.join(
                self.base_path,
                'train' if 'train' in imgid else 'valid',
                imgid + '.jpg',
            )
        else:
            img_path = os.path.join(
                self.base_path,
                imgid + ".jpg",
            )

        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        img = img[None, ...] # add single channel

        if self.transform:
            img = self.transform(img)
                
        sample["pixel_values"] = img

        # text
        sample["labels"] = df_row["caption"]

        # classification
        if hasattr(self, "clf_labels"):
            sample["clf_labels"] = torch.tensor(self.clf_labels[index], dtype=torch.float32)

        return sample

    def __len__(self):
        return len(self.df)



# Class: EvalDataset
class EvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        transform=None,
    ):
        super(EvalDataset, self).__init__()
        self.base_path = base_path
        self.imgs = [r for r in os.listdir(self.base_path) if r.endswith(".jpg") and not r.startswith(".")]

        self.transform = transform

    def __getitem__(self, index):

        imgid = str(self.imgs[index])

        # image
        img_path = os.path.join(
            self.base_path,
            imgid
        )

        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        img = img[None, ...] # add single channel

        if self.transform:
            img = self.transform(img)
                
        pixel_values = img

        return {"id": imgid[:-4], "pixel_values": pixel_values}

    def __len__(self):
        return len(self.imgs)
