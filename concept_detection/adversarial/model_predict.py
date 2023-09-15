# Imports
import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Project Imports
from data_utilities import ImageConceptDataset
from model_utilities import DenseNet121, ResNet50, VGG16



# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default='dataset', help="Directory of the data set.")

# Model
parser.add_argument("--model", type=str, choices=["DenseNet121", "ResNet50", "VGG16"], help="Backbone model to train: DenseNet121, ResNet50 or VGG16.")

# Top-K concepts
parser.add_argument("--top_k_concepts", type=int, default=100, help="Top-K concepts.")

# Subset
parser.add_argument("--subset", type=str, required=True, choices=['train', 'valid', 'test'], help='Subset of the data: train, validation or test.')

# Models directory
parser.add_argument('--modelckpt', type=str, required=True, help="Path to the model weights.")

# Batch size
parser.add_argument('--batchsize', type=int, default=32, help="Batch-size for training and validation.")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")


# Parse the arguments
args = parser.parse_args()

# Data directory
DATA_DIR = args.data_dir

# Subset
SUBSET = args.subset.lower()

# Model
MODEL_NAME = args.model.lower()

# Top-K concepts
TOP_K_CONCEPTS = args.top_k_concepts

# Model Checkpoint
MODELCKPT = args.modelckpt

# Number of workers (threads)
NUM_WORKERS = args.num_workers

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
IMG_CHANNELS = 3
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE


# Evaluation transforms
eval_transforms = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Subset directory
base_datapath = os.path.join(DATA_DIR, 'processed', 'resized')
if SUBSET in ('train', 'valid'):
    test_imgs_dir = None
else:
    test_imgs_dir = os.path.join(base_datapath, 'test')


# Build the evaluation dataset
eval_set = ImageConceptDataset(
    images_dir = base_datapath,
    train_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
    val_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
    subset=SUBSET,
    test_imgs_dir=test_imgs_dir,
    top_k=TOP_K_CONCEPTS,
    transform=eval_transforms,
)


# Dataloaders
eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=NUM_WORKERS)

# Create lists to append batch results
eval_images = []
eval_concepts = []


# Get the number of classes
NR_CLASSES = eval_set.nr_classes
print(f"NR CLASSES {NR_CLASSES}")

# Get some important dictionaries
all_concepts_dict = eval_set.all_concepts_dict



# DenseNet121
if MODEL_NAME == "DenseNet121".lower():
    model = DenseNet121(
        channels=IMG_CHANNELS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        nr_classes=NR_CLASSES
    )

# ResNet18
elif MODEL_NAME == "ResNet50".lower():
    model = ResNet50(
        channels=IMG_CHANNELS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        nr_classes=NR_CLASSES
    )

# VGG16
elif MODEL_NAME == "VGG16".lower():
    model = VGG16(
        channels=IMG_CHANNELS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        nr_classes=NR_CLASSES
    )


# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Weights directory
model_file = os.path.join(MODELCKPT, "weights", "model_best.pt")

# Load model weights
checkpoint = torch.load(model_file, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
print(f"Loaded model from {model_file}")

# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)

# Put model into evaluation mode
model.eval()



# Deactivate gradients
with torch.no_grad():

    # Iterate through dataloader
    for images, labels, img_ids in tqdm(eval_loader):

        # Move data data anda model to GPU (or not)
        images = images.to(DEVICE, non_blocking=True)

        # Get the logits
        logits = model(images)

        # Pass the logits through a Sigmoid to get the outputs
        outputs = torch.sigmoid(logits)
        outputs = outputs.detach().cpu()

        # Add the valid concepts
        for batch_idx in range(images.shape[0]):
            predicted_concepts = ""

            # Get the indices of the predicted concepts (# decision threshold = 0.5)
            indices = torch.where(outputs[batch_idx] >= 0.7)[0].numpy()
            # indices = torch.where(outputs_ >= 0.5)[0].numpy()
            # print(indices)

            for i in indices:
                predicted_concepts += f"{all_concepts_dict[i]};"
                # predicted_concepts += f"{all_concepts_dict[i.item()]};"

            eval_images.append(img_ids[batch_idx].split('.')[0])
            eval_concepts.append(predicted_concepts[:-1])


# Create a dictionary to obtain DataFrame
eval_set = dict()
eval_set["ID"] = eval_images
eval_set["cuis"] = eval_concepts

# Save this into .CSV
evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv(os.path.join(MODELCKPT, f"{SUBSET}_preds.csv"), sep="|", index=False, header=False)
