# Imports
from tqdm import tqdm

# PyTorch Imports
import torch
import torch.nn as nn
from torchvision import models as models

# Project Imports
from asl import AsymmetricLoss, AsymmetricLossOptimized



# Function: Create the model
def model(pretrained, requires_grad, nr_concepts):
    model = models.densenet121(progress=True, pretrained=pretrained)
    # model = models.densenet201(progress=True, pretrained=pretrained)
    # model = models.resnet50(progress=True, pretrained=pretrained)


    for param in model.parameters():
        param.requires_grad = requires_grad

    # make the classification layer learnable
    # we have 8374 classes in total
    # model.fc = nn.Linear(2048, 8374)
    model.classifier = nn.Linear(1024, nr_concepts)
    
    return model



# Function: Create the training loop
def train(model, dataloader, optimizer, criterion, device, weights):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    for data in tqdm(dataloader):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(data)
        if not isinstance(criterion, AsymmetricLoss) and not isinstance(criterion, nn.BCEWithLogitsLoss) and not isinstance(criterion, AsymmetricLossOptimized):
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
        # calculate loss
        loss = criterion(outputs, target)
        if isinstance(criterion, nn.BCELoss):
            loss = (loss * weights).mean()
        # compute gradients
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        train_running_loss += loss.item()

    train_loss = train_running_loss / counter
    
    return train_loss




# Function: Create the validation loop
def validate(model, dataloader, criterion, device, weights):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            # forward pass
            outputs = model(data)
            if not isinstance(criterion, AsymmetricLoss) and not isinstance(criterion, nn.BCEWithLogitsLoss) and not isinstance(criterion, AsymmetricLossOptimized):
                # apply sigmoid activation to get all the outputs between 0 and 1
                outputs = torch.sigmoid(outputs)
            # Calculate loss
            loss = criterion(outputs, target)
            if isinstance(criterion, nn.BCELoss):
                loss = (loss * weights).mean()
            val_running_loss += loss.item()

        val_loss = val_running_loss / counter
        
        return val_loss
