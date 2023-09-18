# Imports
import os
import argparse
from tqdm import tqdm
import pandas as pd

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Transformers Imports
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

# Project Imports
from dataset import EvalDataset


# Run the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to run the script.')

    # Processing parameters
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='Which gpus to use in CUDA_VISIBLE_DEVICES.',
    )
    parser.add_argument(
        '--num_workers', type=int, default=8, help='Number of workers for dataloader.'
    )

    # Directories and paths
    parser.add_argument(
        'ckpt',
        type=str,
        help='Model to be loaded.',
    )
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Directory where data is stored.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        choices=['train', 'valid', 'test'],
        help='Data split for which captions are to be generated.',
    )

    parser.add_argument(
        '--bs',
        type=int,
        default=128,
        help='Batch size.',
    )

    parser.add_argument(
        '--csvfilename',
        type=str,
        default='ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv',
        help="CSV filename for the caption prediction valid labels."
    )



    # Get arguments
    args = parser.parse_args()
    
    # Create checkpoint path
    if args.ckpt[-1] == '/':
        args.ckpt = args.ckpt[:-1]
    save_path = os.path.dirname(args.ckpt)


    # Select device
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    # Build model
    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt)
    model.eval()
    model.to(device)
    feature_extractor = AutoImageProcessor.from_pretrained(os.path.join(save_path, 'feature_extractor'))
    size = feature_extractor.size
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_path, 'tokenizer'))

    # Data
    csvfilename = args.csvfilename
    dtset = EvalDataset(
        os.path.join(args.datadir, args.split),
        img_processor=feature_extractor,
    )
    loader = DataLoader(
        dtset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )


    # Run the algorithm, get predictions and save them into a CSV
    with torch.no_grad():
        res_clef = []

        for sample in tqdm(loader):
            ids = sample['id']
            pixel_values = sample['pixel_values'].to(device)

            output = model.generate(pixel_values=pixel_values)

            pred_str = tokenizer.batch_decode(output, skip_special_tokens=True)

            for i in range(len(ids)):
                res_clef.append({
                    'ID': ids[i],
                    'caption': pred_str[i],
                })

        df = pd.DataFrame(res_clef)
        df.to_csv(os.path.join(save_path, f'{args.split}_preds.csv'), sep='|', index=False, header=None)
