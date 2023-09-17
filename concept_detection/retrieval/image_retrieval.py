# Imports
import os
import argparse
import pandas as pd
import numpy as np

# TensorFlow Imports
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50

# Project Imports
from data_utilities import ImageClefDatasetResNet
from model_image_retrieval_utilities import retrieve



# CLI Interface
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='.', help="Directory of the data set.")
parser.add_argument('--concepts_csv', type=str, default='concepts.csv', help="Csv with id of concepts to be considered.")
parser.add_argument('--num_labels', type=int, default=256, help="Number of labels.")
parser.add_argument('--embed_size', type=int, default=128, help="Size of the embedding.")
parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs.")
parser.add_argument('--opt', type=str, default='Adam', help="Optimiser.")
parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
parser.add_argument('--img_height', type=int, default=224, help="Image height.")
parser.add_argument('--img_wdidth', type=int, default=224, help="Image width.")
parser.add_argument('--img_channels', type=int, default=3, help="Image channels")

# Get the arguments
args = parser.parse_args()
data_dir = args.data_dir
concepts_csv = args.concepts_csv
num_labels = args.num_labels
embed_size = args.embed_size
epochs = args.epochs

# Optimiser and learning rate
opt = args.opt
lr = args.lr
if opt == 'Adam':
    opt = Adam(lr=lr)

# Train shape
train_shape = (args.img_height, args.img_wdidth, args.img_channels)



# Read and load data
print('Reading concepts...')
concepts_filename = os.path.join(data_dir, concepts_csv)
concepts_csv = pd.read_csv(concepts_filename, sep="\t")
all_concepts = concepts_csv["concept"]
concepts = []
dict_concept = dict()
dict_concept_inverse = dict()
for idx, c in enumerate(all_concepts):
    dict_concept[c] = idx
    dict_concept_inverse[idx] = c
    concepts.append(to_categorical(idx, len(all_concepts)))

concepts = np.asarray(concepts)
print(concepts.shape)

# Train data
print('Loading train data...', flush=True)
train_data = ImageClefDatasetResNet(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv',
    'train_resized', 0,
    concepts, num_labels, split=False
)

# Test data
print('Loading test data...', flush=True)
test_data = ImageClefDatasetResNet(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv',
    'valid_resized', 2,
    concepts, num_labels
)



# Build model
model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_tensor=Input(shape=(224, 224, 3)))

# Run retrieval model
retrieve(dataset=test_data, top_images=10, train_data=train_data, model=model, method=1)
