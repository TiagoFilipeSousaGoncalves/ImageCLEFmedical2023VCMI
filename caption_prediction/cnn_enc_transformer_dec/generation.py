# Imports
import os
import argparse
from tqdm import tqdm
import pandas as pd

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
import torchxrayvision as xrv

# Project Imports
from dataset import EvalDataset
from encoder_decoder import CNNTransformerCaptioner



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
        '--csvfilename',
        type=str,
        default='ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv',
        help="CSV filename for the predicted labels."
    )
    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        choices=['train', 'valid', 'test'],
        help='Data split for which captions are to be generated.',
    )

    # Batch size
    parser.add_argument(
        '--bs',
        type=int,
        default=64,
        help='Batch size.',
    )


    # Get arguments
    args = parser.parse_args()

    # Get checkpoint/save path
    if args.ckpt[-1] == '/':
        args.ckpt = args.ckpt[:-1]
    save_path = os.path.dirname(args.ckpt)


    # Get device
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')


    # Build model
    model = CNNTransformerCaptioner.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # Data transforms
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224), xrv.datasets.ToPILImage(), torchvision.transforms.ToTensor()])

    # Data
    csvfilename = args.csvfilename
    dtset = EvalDataset(
        os.path.join(args.datadir, args.split),
        transform=transform,
    )
    loader = DataLoader(
        dtset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )


    # Predict labels
    with torch.no_grad():
        res_clef = []

        for sample in tqdm(loader):
            ids = sample['id']
            pixel_values = sample['pixel_values'].to(device)

            pred_str = model.generate(pixel_values=pixel_values)

            for i in range(len(ids)):
                res_clef.append({
                    'ID': ids[i],
                    'caption': pred_str[i],
                })

        df = pd.DataFrame(res_clef)
        df.to_csv(os.path.join(save_path, f'{args.split}_preds.csv'), sep='|', index=False, header=None)
