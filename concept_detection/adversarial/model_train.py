# Imports
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import datetime

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Weights and Biases (W&B) Imports
import wandb

# Log in to W&B Account
wandb.login()

# Project Imports
from data_utilities import ImageConceptDataset
from model_utilities import DenseNet121, ResNet50, VGG16, ConceptDiscriminator

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument("--data_dir", type=str, default="dataset", help="Directory of the data set.")

# Model
parser.add_argument("--model", type=str, choices=["DenseNet121", "ResNet50", "VGG16"], help="Backbone model to train: DenseNet121, ResNet50 or VGG16.")

# Top-K concepts
parser.add_argument("--top_k_concepts", type=int, default=100, help="Top-K concepts.")

# Class weights
parser.add_argument("--class_weights", action="store_true", help="Compute class weights for the loss.")

# Batch size
parser.add_argument("--batchsize", type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument("--imgsize", type=int, default=224, help="Size of the image after transforms")

# Number of epochs
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")

# Learning rate
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--results_dir", type=str, default="results/concept_detection/adversarial", help="Results directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Save frequency
parser.add_argument("--save_freq", type=int, default=10, help="Frequency (in number of epochs) to save the model")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint from which to resume training")


# Parse the arguments
args = parser.parse_args()

# Resume training
if args.resume:
    assert args.ckpt is not None, "Please specify the model checkpoint when resume is True"

RESUME = args.resume

# Training checkpoint
CKPT = args.ckpt

# Data directory
DATA_DIR = args.data_dir

# Results directory
RESULTS_DIR = args.results_dir

# Number of workers (threads)
NUM_WORKERS = args.num_workers

# Number of training epochs
EPOCHS = args.epochs

# Learning rate
LEARNING_RATE = args.lr

# Class weights
USE_CLASSWEIGHTS = args.class_weights

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Save frquency
SAVE_FREQ = args.save_freq

# Top-K concepts
TOP_K_CONCEPTS = args.top_k_concepts

# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join(RESULTS_DIR, timestamp)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Save training parameters
with open(os.path.join(RESULTS_DIR, "train_params.txt"), "w") as f:
    f.write(str(args))



# Set the W&B project
wandb.init(
    project="ImageCLEF2023medical", 
    name=timestamp,
    config={
        "name":"adversarial",
        "model": args.model,
        "nr_epochs": EPOCHS,
    }
)



# Results and Weights
weights_dir = os.path.join(RESULTS_DIR, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)

# History Files
history_dir = os.path.join(RESULTS_DIR, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)

# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, "tensorboard"), flush_secs=5)


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



# Load data
# Train
# Transforms
train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, fill=(0, 0, 0)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation
# Transforms
valid_transforms = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Train set
train_set = ImageConceptDataset(
    images_dir = os.path.join(DATA_DIR, 'processed', 'resized'),
    train_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
    val_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
    subset='complete',
    top_k=TOP_K_CONCEPTS, 
    transform=train_transforms
)

# Validation set
valid_set = ImageConceptDataset(
    images_dir = os.path.join(DATA_DIR, 'processed', 'resized'),
    train_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
    val_concept_detection_csv=os.path.join(DATA_DIR, 'original', 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
    subset='valid',
    top_k=TOP_K_CONCEPTS,
    transform=valid_transforms
)



# Get the number of classes
NR_CLASSES = train_set.nr_classes



# Choose model(s)
classifier_name = args.model.lower()

# DenseNet121
if classifier_name == "DenseNet121".lower():
    classifier = DenseNet121(
        channels=IMG_CHANNELS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        nr_classes=NR_CLASSES
    )

# ResNet18
elif classifier_name == "ResNet50".lower():
    classifier = ResNet50(
        channels=IMG_CHANNELS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        nr_classes=NR_CLASSES
    )

# VGG16
elif classifier_name == "VGG16".lower():
    classifier = VGG16(
        channels=IMG_CHANNELS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        nr_classes=NR_CLASSES
    )


# Put model into DEVICE (CPU or GPU)
classifier = classifier.to(DEVICE)


# Initialise Discriminator
discriminator = ConceptDiscriminator(input_size=train_set.nr_classes)
discriminator = discriminator.to(DEVICE)


# Log model's parameters and gradients to W&B
wandb.watch(classifier)
wandb.watch(discriminator)


# Compute loss weights
if USE_CLASSWEIGHTS:
    pos_count = np.count_nonzero(train_set.labels.copy(), axis=0)
    neg_count = len(train_set.labels.copy()) - pos_count
    np.testing.assert_array_equal(np.ones_like(pos_count) * len(train_set.labels.copy()), np.sum((neg_count, pos_count), axis=0))
    pos_weights = neg_count / pos_count
else:
    pos_weights = None



# Hyper-parameters for the classifier
LOSS = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.from_numpy(pos_weights).to(DEVICE))
VAL_LOSS = torch.nn.BCEWithLogitsLoss(reduction="mean")
OPTIMISER = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)


# Hyperparameters for the discriminator
LOSS_DISCRIMINATOR = torch.nn.BCEWithLogitsLoss(reduction="mean")
REAL_LABEL = 1.
FAKE_LABEL = 0.
OPTIMISER_DISCRIMINATOR = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Resume training from given checkpoint
if RESUME:
    checkpoint = torch.load(CKPT)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    OPTIMISER.load_state_dict(checkpoint["optimizer_state_dict"])
    init_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from {CKPT} at epoch {init_epoch}")
else:
    init_epoch = 0



# Dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Go through the number of Epochs
for epoch in range(init_epoch, EPOCHS):
    # Epoch
    print(f"Epoch: {epoch+1}")

    # Training Loop
    print("Training Phase")

    # Running train losses
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)
    run_train_Dloss = torch.tensor(0, dtype=torch.float64, device=DEVICE)
    run_train_Gloss = torch.tensor(0, dtype=torch.float64, device=DEVICE)

    # Put classifier and discriminator in training mode
    classifier.train()
    discriminator.train()


    # Iterate through dataloader
    for images, labels, _ in tqdm(train_loader):

        # Move data and model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)


        # STEP 1
        # Train the classifier
        OPTIMISER.zero_grad(set_to_none=True)

        # Get logits
        labels_pred = classifier(images)

        # Compute the batch loss
        loss = LOSS(labels_pred, labels)

        # Update batch losses
        run_train_loss += loss.item()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        OPTIMISER.step()



        # STEP 2
        # We provide the Discriminator with an all-real batch
        OPTIMISER_DISCRIMINATOR.zero_grad(set_to_none=True)

        # Get a batch of real labels
        real_label_batch = torch.full((labels.size(0),), REAL_LABEL, dtype=torch.float, device=DEVICE)

        # Forward pass real batch through D
        disc_output = discriminator(labels).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = LOSS_DISCRIMINATOR(disc_output, real_label_batch)
        
        # Calculate gradients for D in backward pass
        errD_real.backward()



        # STEP 3
        # We provide the Discriminator with an all-fake batch (i.e., the concept predictions by the classifier)
        # Get the concept predictions by the classifier
        labels_pred = classifier(images)

        # Fake label batch for the discriminator
        fake_label_batch = torch.full((labels.size(0),), FAKE_LABEL , dtype=torch.float, device=DEVICE)

        # Classify all fake batch with D
        disc_output = discriminator(labels_pred.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = LOSS_DISCRIMINATOR(disc_output, fake_label_batch)
        
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update Dloss
        run_train_Dloss += errD.item()
        
        # Update Discriminator
        OPTIMISER_DISCRIMINATOR.step()


        # STEP 4
        # Optimise classifier according to discriminator output
        OPTIMISER.zero_grad()
        real_label_batch = torch.full((labels.size(0),), REAL_LABEL, dtype=torch.float, device=DEVICE)
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        disc_output = discriminator(labels_pred).view(-1)
        
        # Calculate classifier's loss based on this output
        err_clf = LOSS_DISCRIMINATOR(disc_output, real_label_batch)

        # Update train Gloss
        run_train_Gloss += err_clf.item()
        
        # Calculate gradients for G
        err_clf.backward()
        
        # Update classifier again
        OPTIMISER.step()



    # Compute Average Train Loss
    avg_train_loss = run_train_loss / len(train_loader)
    avg_train_Dloss = run_train_Dloss / len(train_loader)
    avg_train_Gloss = run_train_Gloss / len(train_loader)

    # Print Statistics
    print(f"Train Loss: {avg_train_loss.item()}")
    print(f"Train D-Loss: {avg_train_Dloss.item()}")
    print(f"Train G-Loss: {avg_train_Gloss.item()}")

    # Plot to Tensorboard and W&B
    wandb_tr_metrics = dict()
    wandb_tr_metrics["loss/train"] = avg_train_loss.item()
    wandb_tr_metrics["loss/d-train"] = avg_train_Dloss.item()
    wandb_tr_metrics["loss/g-train"] = avg_train_Gloss.item()
    
    # Log to W&B
    wandb.log(wandb_tr_metrics)


    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss



    # Validation Loop
    print("Validation Phase")

    # Running train losses
    run_val_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)

    # Put model in evaluation mode
    classifier.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels, _ in tqdm(val_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = classifier(images)

            # Compute the batch loss
            loss = VAL_LOSS(logits, labels)

            # Update batch losses
            run_val_loss += loss.item()

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss / len(val_loader)

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss.item()}")

        # Plot to Tensorboard and W&B
        wandb_val_metrics = dict()
        wandb_val_metrics["loss/val"] = avg_val_loss.item()
        wandb.log(wandb_val_metrics)


        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"model_best.pt")

            save_dict = {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": OPTIMISER.state_dict(),
                "loss": avg_train_loss,
            }
            torch.save(save_dict, model_path)

            print(f"Successfully saved at: {model_path}")

        # Checkpoint loop/condition
        if epoch % SAVE_FREQ == 0 and epoch > 0:

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"model_{epoch:04}.pt")

            save_dict = {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": OPTIMISER.state_dict(),
                "loss": avg_train_loss,
            }
            torch.save(save_dict, model_path)



# Finish statement and W&B
wandb.finish()
print("Finished.")
