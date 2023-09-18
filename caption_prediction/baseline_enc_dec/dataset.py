# Imports
import os
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch



# Class: CustomDataCollator
class CustomDataCollator:
    def __init__(self, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        imgs = [i['pixel_values'] for i in batch]
        imgs = torch.stack(imgs)

        # from the VisionEncoderDecoder documentation:
        # For training, decoder_input_ids are automatically created by the model by shifting the labels to the right, replacing -100 by the pad_token_id and prepending them with the decoder_start_token_id.
        captions = [i['labels'] + self.tokenizer.eos_token for i in batch]
        captions = self.tokenizer(
            captions,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        captions = torch.where(captions.attention_mask == 0, -100, captions.input_ids) # ignore padding tokens in loss

        out_batch = {
            'pixel_values': imgs,
            'labels': captions
        }

        return out_batch



# ClasS: Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        gt_file,
        img_processor=None,
    ):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.trainval = 'trainval' in gt_file
        self.df = pd.read_csv(gt_file, sep='\t')
        #self.df = self.df[:500] 

        self.img_processor = img_processor

    def __getitem__(self, index):

        df_row = self.df.iloc[index]

        sample = {}

        imgid = str(df_row['ID'])

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
                imgid + '.jpg',
            )

        img = Image.open(img_path).convert('RGB')

        if self.img_processor:
            img = self.img_processor(img, return_tensors="pt")
                
        sample['pixel_values'] = img['pixel_values'].squeeze(0)

        # text
        # from the VisionEncoderDecoder documentation:
        # For training, decoder_input_ids are automatically created by the model by shifting the labels to the right, replacing -100 by the pad_token_id and prepending them with the decoder_start_token_id.
        sample['labels'] = df_row['caption']
        
        return sample

    def __len__(self):
        return len(self.df)



# Class: EvalDataset
class EvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        img_processor=None,
    ):
        super(EvalDataset, self).__init__()
        self.base_path = base_path
        self.imgs = [r for r in os.listdir(self.base_path) if r.endswith('.jpg') and not r.startswith('.')]

        self.img_processor = img_processor

    def __getitem__(self, index):

        imgid = str(self.imgs[index])

        # image
        img_path = os.path.join(
            self.base_path,
            imgid
        )

        img = Image.open(img_path).convert('RGB')

        if self.img_processor:
            img = self.img_processor(img, return_tensors="pt")
                
        pixel_values = img['pixel_values'].squeeze(0)

        return {'id': imgid[:-4], 'pixel_values': pixel_values}

    def __len__(self):
        return len(self.imgs)
