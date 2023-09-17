# Imports
import os
import argparse
import numpy as np
import pandas as pd
import shutil

# TensorFlow Imports
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Project Imports
from data_utilities import ImageClefDataset
from model_retrieval_utilities import build_feature_extractor, build_label_encoder, distance_lambda, contrastive_loss, get_threshold



# CLI Interface
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='.', help="Directory of the data set.")
parser.add_argument('--iti', action='store_true', help="Image to image distance is minimized in loss function.")
parser.add_argument('--ctc', action='store_true', help="Concept to concept distance is minimized in loss function.")
parser.add_argument('--concepts_csv', type=str, default='concepts.csv', help="Csv with id of concepts to be considered.")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and validation.")
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs for training.")
parser.add_argument('--workers', type=int, default=4, help="Number of workers for data loader.")
parser.add_argument('--infer', action='store_true', help="Model is not trained.")
parser.add_argument('--path_weights', type=str, default='models/weights_1000.h5', help="Path to model weights to load on inference.")
parser.add_argument('--num_labels', type=int, default=256, help="Number of labels.")
parser.add_argument('--embed_size', type=int, default=128, help="Size of the embedding.")
parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs.")
parser.add_argument('--opt', type=str, default='Adam', help="Optimiser.")
parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
parser.add_argument('--img_height', type=int, default=224, help="Image height.")
parser.add_argument('--img_wdidth', type=int, default=224, help="Image width.")
parser.add_argument('--img_channels', type=int, default=1, help="Image channels")
parser.add_argument('--loss_type', type=str, default='itc', choices=['itc', 'iti', 'ctc', 'all'], help="Loss type.")

# Get the arguments
args = parser.parse_args()
data_dir = args.data_dir
concepts_csv = args.concepts_csv
num_labels = args.batch_size
epochs = args.epochs
workers = args.workers
infer = args.infer
model_weights = args.path_weights
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

# Loss type
loss_type = args.loss_type
if args.iti and args.ctc:
    assert loss_type == 'all'
elif args.iti:
    assert loss_type == 'iti'
elif args.ctc:
    assert loss_type == 'ctc'



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
train_data = ImageClefDataset(
    dict_concept,
    os.path.join(data_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
    os.path.join(data_dir, 'train'), 0,
    concepts, num_labels, losses=loss_type
)

# Validation data
print('Loading valid data...', flush=True)
valid_data = ImageClefDataset(
    dict_concept,
    os.path.join(data_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv'),
    os.path.join(data_dir, 'train'), 1,
    concepts, num_labels, losses=loss_type, split=True
)

# Test data
print('Loading test data...', flush=True)
test_data = ImageClefDataset(
    dict_concept,
    os.path.join(data_dir, 'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv'),
    os.path.join(data_dir, 'valid'), 2,
    concepts, 1, losses=loss_type
)



# Build model
input_imgs = Input(train_shape)
feature_extractor = build_feature_extractor(train_shape, embed_size)
label_encoder = build_label_encoder(all_concepts, embed_size)

input_img = Input(train_shape)
input_label = Input((len(all_concepts),))
feat_img = feature_extractor(input_img)
feat_label = label_encoder(input_label)

distance_itc = Lambda(lambda x: distance_lambda(x[0], x[1]), output_shape=(num_labels,))([feat_img, feat_label])
if loss_type == 'iti' or loss_type == 'all':
    distance_iti = Lambda(lambda x: distance_lambda(x[0], x[1]), output_shape=(num_labels,))([feat_img, feat_img])
if loss_type == 'ctc' or loss_type == 'all':
    distance_ctc = Lambda(lambda x: distance_lambda(x[0], x[1]), output_shape=(num_labels,))([feat_label, feat_label])

if loss_type == 'iti':
    training_model = Model([input_img, input_label], [distance_itc, distance_iti])
elif loss_type == 'ctc':
    training_model = Model([input_img, input_label], [distance_itc, distance_ctc])
elif loss_type == 'all':
    training_model = Model([input_img, input_label], [distance_itc, distance_ctc. distance_iti])
else:
    training_model = Model([input_img, input_label], distance_itc)
training_model.compile(optimizer=opt, loss=contrastive_loss)
training_model.summary()



# Get checkpoint path and fit the model
path = os.path.join(data_dir, 'models')
if infer:
    training_model.load_weights(model_weights)
else:
    if os.path.isdir(path) == True:
        shutil.rmtree(path)
        os.makedirs(path)

    checkpoint = ModelCheckpoint( os.path.join(data_dir, 'weights_{epoch:03d}.h5'), period=50) 
    training_model.fit_generator(train_data, validation_data=valid_data, callbacks=[checkpoint], epochs=epochs, steps_per_epoch=len(train_data), workers=workers)



# Predict labels
labels = range(0, len(all_concepts))
labels = to_categorical(labels, len(all_concepts))
encoded_labels = label_encoder.predict(labels)
threshold = get_threshold(valid_data, feature_extractor, encoded_labels, all_concepts)
infer(threshold, test_data, feature_extractor, encoded_labels, all_concepts, dict_concept_inverse)
