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
# from torchvision.models import densenet121, resnet18, DenseNet121_Weights, ResNet18_Weights
from torchvision.models import densenet121, resnet18

# Project Imports
from data_utilities import ImgClefConcDataset



# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default='data', help="Directory of the data set.")

# Model
parser.add_argument('--model', type=str, choices=["DenseNet121", "ResNet18"], default="DenseNet121", help="Baseline model (DenseNet121, ResNet18).")

# Semantic type
parser.add_argument("--semantic_type", type=str, required=True, choices=['Body Part, Organ, or Organ Component', 'Disease or Syndrome', 'Diagnostic Procedure', 'Body Location or Region', 'Pathologic Function', 'Body Space or Junction', 'Neoplastic Process', 'Congenital Abnormality', 'Anatomical Abnormality', 'Medical Device', 'Functional Concept', 'Tissue', 'Body Substance', 'Acquired Abnormality', 'Qualitative Concept', 'Body System', 'Organ or Tissue Function', 'Organism Function', 'Manufactured Object', 'Spatial Concept', 'Substance'], help="Semantic type: 'Body Part, Organ, or Organ Component', 'Disease or Syndrome', 'Diagnostic Procedure', 'Body Location or Region', 'Pathologic Function', 'Body Space or Junction', 'Neoplastic Process', 'Congenital Abnormality', 'Anatomical Abnormality', 'Medical Device', 'Functional Concept', 'Tissue', 'Body Substance', 'Acquired Abnormality', 'Qualitative Concept', 'Body System', 'Organ or Tissue Function', 'Organism Function', 'Manufactured Object', 'Spatial Concept', 'Substance'.")

# Subset
parser.add_argument("--subset", type=str, required=True, choices=['train', 'validation', 'test'], help='Subset of the data: train, validation or test.')

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

# Semantic type
SEMANTIC_TYPE = args.semantic_type

# Subset
SUBSET = args.subset.lower()

# Model
MODEL_NAME = args.model.lower()

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
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Evaluation transforms
eval_transforms = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Subset directory
base_datapath = os.path.join(DATA_DIR, 'processed', 'resized')
if SUBSET == 'train':
    img_datapath = os.path.join(base_datapath, 'train')
    test_imgs_dir = None

elif SUBSET == 'validation':
    img_datapath = os.path.join(base_datapath, 'valid')
    test_imgs_dir = None

else:
    img_datapath = os.path.join(base_datapath, 'test')
    test_imgs_dir = img_datapath


# Build the evaluation dataset
eval_set = ImgClefConcDataset(
    img_datapath=img_datapath,
    cui_semantic_types_csv=os.path.join(DATA_DIR, 'processed', "ImageCLEFmedical_Caption_2023_cui_semantic_types.csv"),
    semantic_type=SEMANTIC_TYPE,
    train_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
    val_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
    test_imgs_dir=test_imgs_dir,
    transform=eval_transforms,
    subset=SUBSET,
    classweights=None
)


# Dataloaders
eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=NUM_WORKERS)

# Create lists to append batch results
eval_images = []
eval_concepts = []



# Perform inference
print(f"Loading the semantic type <{SEMANTIC_TYPE}> from {MODELCKPT}")

# Get the number of classes
NR_CLASSES = eval_set.nr_classes
print(f"NR CLASSES {NR_CLASSES}")

# Get some important dictionaries
inv_sem_type_concepts_dict = eval_set.inv_sem_type_concepts_dict



# Load model(s)
# DenseNet121
if MODEL_NAME == "densenet121":
    # model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model = densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(1024, NR_CLASSES)

# ResNet18
elif MODEL_NAME == "resnet18":
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, NR_CLASSES)


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
            indices = torch.where(outputs[batch_idx] >= 0.5)[0].numpy()
            for i in indices:
                predicted_concepts += f"{inv_sem_type_concepts_dict[i]};"

            eval_images.append(img_ids[batch_idx])
            eval_concepts.append(predicted_concepts[:-1])


# Create a dictionary to obtain DataFrame
eval_set = dict()
eval_set["ID"] = eval_images
eval_set["cuis"] = eval_concepts

# Save this into .CSV
evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv(os.path.join(MODELCKPT, f"{SUBSET}_preds.csv"), sep="|", index=False, header=False)
