# Imports
import argparse
import os
from tqdm import tqdm

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Transformers Imports
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Sklearn Imports
import sklearn.metrics

# Project Imports
from dataset import Dataset, CustomDataCollator



# Run the script
if __name__ == '__main__':

    # CLI Arguments
    parser = argparse.ArgumentParser(description='Arguments to run the script.')

    # GPU
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='Which gpus to use in CUDA_VISIBLE_DEVICES.',
    )

    # Number of Workers
    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of workers for dataloader.'
    )

    # Directories and paths
    parser.add_argument(
        '--basedir',
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

    # Hyperparameters
    parser.add_argument(
        '--bs',
        type=int,
        default=1,
        help='Batch size.',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Max length of inputs.',
    )


    # Get parsed arguments
    args = parser.parse_args()


    # Select device
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    # Build model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained("results/2023-04-27_15-10-04/checkpoint-best").to(device)
    model.eval()


    # Get data
    NR_CONCEPTS = 2125
    BASE_DIR = args.basedir

    if NR_CONCEPTS == 100:
        concept_dict = os.path.join(BASE_DIR, "new_top100_concepts.csv")
    else:
        concept_dict = os.path.join(BASE_DIR, "concepts.csv")

    # Build Dataset
    dtset = Dataset(
        gt_file=os.path.join(args.basedir, 'captions_test.csv'),
        concepts_file=os.path.join(args.basedir, 'concepts_test.csv'),
        df_all_concepts=concept_dict
    )

    # Data collator
    collator = CustomDataCollator(tokenizer, args.max_length)

    # Dataloader
    loader = DataLoader(
        dtset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )


    # Perform inference
    with torch.no_grad():
        y_true = []
        y_pred = []

        for sample in tqdm(loader):
            gt_concepts = sample['labels']
            y_true.append(gt_concepts[0].numpy())
            caption = sample['input_ids'].to(device)

            output = model(caption).logits
            y_pred.append(torch.where(output > 0, 1, 0)[0].to(torch.float).cpu().numpy())


    # Generate evaluation report
    print(f"/////////// Evaluation Report ////////////")
    print(f"Exact Match Ratio: {sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
    print(f"Hamming loss: {sklearn.metrics.hamming_loss(y_true, y_pred):.4f}")
    print(f"Recall: {sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
    print(f"Precision: {sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
    print(f"F1 Measure: {sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
