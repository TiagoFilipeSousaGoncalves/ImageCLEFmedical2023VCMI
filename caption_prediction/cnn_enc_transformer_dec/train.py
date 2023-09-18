# Imports
import os
import argparse
import datetime
import random
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
import torchxrayvision as xrv

# PyTorch Lightning Imports
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

# Transformers Imports
from transformers import AutoTokenizer

# Project Imports
from dataset import Dataset, CustomDataCollator
from encoder_decoder import CNNTransformerCaptioner



# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(0)



# Function: Create folder(s)
def _create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(folder, timestamp)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return timestamp, results_path



# Function: Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad), sum(p.numel() for p in model.parameters()
                                        if not p.requires_grad)



# Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to run the script.")

    # Processing parameters
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Which gpus to use in CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader."
    )

    # Directories and paths
    parser.add_argument(
        "--logdir",
        type=str,
        default="results",
        help="Directory where logs and models are to be stored.",
    )
    parser.add_argument(
        "--basedir",
        type=str,
        required=True,
        help="Directory where data is stored.",
    )

    # Model
    parser.add_argument(
        "--custom_tokenizer",
        type=str,
        default=None,
        help="Use custom tokenizer trained on the dataset. If the tokenizer from the model is to be used leave None.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="distilgpt2",
        choices=[
            "bert-base-uncased",
            "gpt2",
            "distilgpt2",
        ],
        help="Decoder model to load.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze the encoder.",
    )
    parser.add_argument(
        "--clf",
        type=str,
        default=None,
        help="Path to caption2concept classifier. If the path is set, an RL classification loss will be added.",
    )
    parser.add_argument(
        "--clf_weight",
        type=float,
        default=0.1,
        help="Weight for classification loss.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Max length of inputs.",
    )

    # Training
    parser.add_argument(
        '--trainval',
        action='store_true',
        help='Train model on trainval split (for final submissions).'
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Load model from this checkpoint."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=2000,
        help="Frequency of validation."
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="Learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Logging
    parser.add_argument(
        "--wandb_online",
        action="store_true",
        help="Start WANDB in online sync mode.",
    )

    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )


    # Get the arguments
    args = parser.parse_args()
    
    # Select device and tune some device parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")


    # Create checkpoint folder
    timestamp, path = _create_folder(args.logdir)
    
    # Save experiment parameters
    with open(os.path.join(path, "train_params.txt"), "w") as f:
        f.write(str(args))


    # Build model
    if args.custom_tokenizer:
        tokenizer_path = args.custom_tokenizer
    else:
        tokenizer_path = args.decoder

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(os.path.join(path, "tokenizer"))


    # Data transforms
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224), xrv.datasets.ToPILImage(), torchvision.transforms.ToTensor()])

    # Get concepts CSVs
    concepts_file = None
    df_all_concepts = None
    if args.clf:
        concepts_file = os.path.join(args.basedir, "ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv")
        df_all_concepts = os.path.join(args.basedir, "unique_concepts.csv")
    
    # Data
    if args.trainval:
        if concepts_file is not None:
            concepts_file = concepts_file.replace("train", "trainval")
        tr_dtset = Dataset(
            os.path.join(args.basedir),
            os.path.join(args.basedir, 'ImageCLEFmedical_Caption_2023_caption_prediction_trainval_labels.csv'),
            concepts_file=concepts_file,
            df_all_concepts=df_all_concepts,
            transform=transform,
        )
    else:
        tr_dtset = Dataset(
            os.path.join(args.basedir, "train"),
            os.path.join(args.basedir, "ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv"),
            concepts_file=concepts_file,
            df_all_concepts=df_all_concepts,
            transform=transform,
        )
    val_dtset = Dataset(
        os.path.join(args.basedir, "valid"),
        os.path.join(args.basedir, "ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv"),
        concepts_file=concepts_file.replace("train", "valid") if concepts_file is not None else concepts_file,
        df_all_concepts=df_all_concepts,
        transform=transform,
    )
    collator = CustomDataCollator(tokenizer, args.max_length)
    tr_loader = DataLoader(
        tr_dtset, 
        batch_size=args.bs,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dtset, 
        batch_size=args.bs,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        shuffle=False,
    )

    # Model
    model = CNNTransformerCaptioner(
        decoder=args.decoder,
        freeze_encoder=args.freeze_encoder,
        clf=args.clf,
        clf_weight=args.clf_weight,
        tokenizer=tokenizer,
        max_length=args.max_length,
        lr=args.lr,
        min_lr=args.min_lr,
        train_steps=args.max_steps
    )


    # Make parameters trainable (or not)
    trainable_params, non_trainable_params = count_parameters(model.encoder)
    print("Encoder params\n\t Trainable: %d \n\t Non trainable: %d\n" % (trainable_params, non_trainable_params))

    trainable_params, non_trainable_params = count_parameters(model.decoder)
    print("Decoder params\n\t Trainable: %d \n\t Non trainable: %d\n" % (trainable_params, non_trainable_params))


    # Setup WandB
    wandb_logger = WandbLogger(
        project="imageclef23_cpt",
        config=args,
        name=timestamp,
        save_dir=path,
        offline=not args.wandb_online,
    )


    # Create model checkpoint
    checkpoint_callback_best_loss = ModelCheckpoint(
        dirpath=path, 
        filename="checkpoint_best_{val_loss:.5f}",
        verbose=True,
        save_top_k=1, 
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    checkpoint_callback_best_metric = ModelCheckpoint(
        dirpath=path, 
        filename="checkpoint_best_{val_bert_score:.5f}",
        verbose=True,
        save_top_k=1, 
        monitor="val_bert_score",
        mode="max",
    )
    early_stopping = EarlyStopping(monitor="val_bert_score", patience=args.patience, verbose=True, mode="max", strict=True)
    lr_monitor = LearningRateMonitor()


    # Debuggin mode (or not)
    if args.debug:
        limit_train_batches = 20
        limit_val_batches = 20
        args.eval_steps = 30
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0


    # Train model
    trainer = L.Trainer(
        devices=-1,  # use CUDA_VISIBLE_DEVICES to choose devices
        logger=wandb_logger,
        callbacks=[checkpoint_callback_best_loss, checkpoint_callback_best_metric, early_stopping, lr_monitor],
        limit_train_batches=limit_train_batches, # useful for debugging,
        limit_val_batches=limit_val_batches, # useful for debugging,
        max_steps=args.max_steps,
        val_check_interval=args.eval_steps,
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=path,
    )
    trainer.fit(model, tr_loader, val_loader, ckpt_path=args.ckpt)
