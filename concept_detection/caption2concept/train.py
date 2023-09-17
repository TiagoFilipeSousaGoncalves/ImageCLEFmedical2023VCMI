# Imports
import os
import argparse
import datetime
import numpy as np
import shutil
import random

# PyTorch Imports
import torch

# Sklearn Imports
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Transformers Imports
from transformers import (
    AutoTokenizer,    
    DistilBertForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    EvalPrediction
)

# Project Imports
from dataset import Dataset, CustomDataCollator



# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(0)



# Function: Multi-label metrics
# Source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    
    # First, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # Next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    
    # Finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Return as dictionary
    metrics = {
        'f1': f1_micro_average,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }
    
    return metrics



# Function: Compute metrics
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids
    )
    
    return result



# Function: Create a folder
def _create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(folder, timestamp)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return timestamp, results_path



# Function: Count parameters
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters() if not p.requires_grad)



# Run train loop
if __name__ == '__main__':

    # CLI Arguments
    parser = argparse.ArgumentParser(description='Arguments to run the script.')

    # GPU Device
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='Which gpus to use in CUDA_VISIBLE_DEVICES.',
    )

    # Number of Workers
    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
        help='Number of workers for dataloader.'
    )

    # FP Precision
    parser.add_argument(
        '--fp16',
        dest='fp16',
        action='store_true',
        help='Use 16-bit floating-point precision.',
    )

    # No FP Precision
    parser.add_argument(
        '--no-fp16',
        dest='fp16',
        action='store_false',
        help='Use 16-bit floating-point precision.',
    )
    parser.set_defaults(fp16=False)

    # Directories and paths
    parser.add_argument(
        '--logdir',
        type=str,
        default='results',
        help='Directory where logs and models are to be stored.',
    )
    parser.add_argument(
        '--basedir',
        type=str,
        required=True,
        help='Directory where data is stored.',
    )

    # Model
    parser.add_argument(
        '--custom_tokenizer',
        type=str,
        default=None,
        help='Use custom tokenizer trained on the dataset. If the tokenizer from the model is to be used leave None.',
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='facebook/deit-tiny-distilled-patch16-224',
        choices=[
            'microsoft/beit-base-patch16-224-pt22k-ft22k',
            'google/vit-base-patch16-224', 'facebook/deit-tiny-patch16-224', 'facebook/deit-tiny-distilled-patch16-224'
        ],
        help='Encoder model to load.',
    )
    parser.add_argument(
        '--decoder',
        type=str,
        default='distilbert-base-uncased',
        choices=[
            'bert-base-uncased',
            'gpt2',
            'distilgpt2',
        ],
        help='Decoder model to load.',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Max length of inputs.',
    )

    # Training
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='Load model from this checkpoint.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint.'
    )
    parser.add_argument(
        '--load_pretrained',
        action='store_true',
        help='Load pre trained model for fine tuning.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs.'
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate.',
    )
    parser.add_argument(
        '--decay',
        type=float,
        default=1e-6,
        help='Learning rate.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience',
    )

    # Get arguments
    args = parser.parse_args()
    if args.resume or args.load_pretrained:
        assert args.resume != args.load_pretrained, 'Options resume and load_pretrained are mutually exclusive. Please choose only one.'
    if args.ckpt:
        assert (
                args.resume or args.load_pretrained
        ), 'When resuming training or loading pretrained model, you need to provide a checkpoint. When a checkpoint is provided, you need to select either the resume or the load_pretrained options.'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # Device
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    # Create checkpoint directory
    timestamp, path = _create_folder(args.logdir)

    # Save training parameters
    with open(os.path.join(path, 'train_params.txt'), 'w') as f:
        f.write(str(args))

    # WandB Config
    os.environ['WANDB_PROJECT'] = 'imageclef23_cpt'
    os.environ['WANDB_DIR'] = path


    # Build model
    if args.ckpt:
        tokenizer_path = args.ckpt
    else:
        if args.custom_tokenizer:
            tokenizer_path = args.custom_tokenizer
        else:
            tokenizer_path = args.decoder
    feature_extractor_path = args.ckpt if args.ckpt else args.encoder

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Read the training .CSV file
    NR_CONCEPTS = 2125
    BASE_DIR = args.basedir

    if (NR_CONCEPTS == 100):
        concept_dict = os.path.join(BASE_DIR, "new_top100_concepts.csv")
    else:
        concept_dict = os.path.join(BASE_DIR, "concepts.csv")

    # Data
    tr_dtset = Dataset(
        gt_file=os.path.join(args.basedir, 'ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv'),
        concepts_file=os.path.join(args.basedir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
        df_all_concepts=concept_dict
    )

    val_dtset = Dataset(
        gt_file=os.path.join(args.basedir, 'captions_valid.csv'),
        concepts_file=os.path.join(args.basedir, 'concepts_valid.csv'),
        df_all_concepts=concept_dict
    )

    # Model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", problem_type="multi_label_classification", num_labels=NR_CONCEPTS)

    # Trainer
    training_args = TrainingArguments(
        output_dir=path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        weight_decay=args.decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        fp16=args.fp16,
        dataloader_drop_last=True,
        dataloader_num_workers=args.num_workers,
        run_name=timestamp,
        report_to="wandb"
    )

    # Data collator
    collator = CustomDataCollator(tokenizer, args.max_length)

    # Initialise Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_dtset,
        eval_dataset=val_dtset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(args.patience)]
    )

    # Train
    trainer.train()

    # Save best model
    print('Saving best model...')
    best_model_dir = trainer.state.best_model_checkpoint
    shutil.copytree(best_model_dir, os.path.join(path, 'checkpoint-best'))
