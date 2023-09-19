# Imports
import os
import argparse
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm

# Define Keras Backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Tensorflow & Keras Imports
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

# Project Imports
from load_autoregressive_data import ImageClefDataset
from model_utilities import build_classification_model, make_trainable, show_metrics



# Command Line Interface
parser = argparse.ArgumentParser()
parser.add_argument("--num_labels", type=int, default=256, help="Number of labels.")
parser.add_argument("--embed_size", type=int, default=2048, help="Size of the embedding.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
parser.add_argument("--opt", type=str, default='Adam', choices=['Adam'], help="Optimiser.")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
parser.add_argument("--img_height", type=int, default=224, help="Image height.")
parser.add_argument("--img_width", type=int, default=224, help="Image width.")
parser.add_argument("--img_channels", type=int, default=3, help="Image channels.")
parser.add_argument("--num_layers", type=int, default=17, help="Number of classification layers for the autoregressive model.")
parser.add_argument("--num_units", type=int, default=125, help="Number of units for the dense layers.")
parser.add_argument("--concepts_filename", type=str, default='top2125_concepts.csv', help="The concepts filename.")
parser.add_argument("--checkpoint_path", type=str, default= "./models_vgg", help="The path for the checkpoints.")
args = parser.parse_args()



# Training hyperparameters
num_labels = args.num_labels
embed_size = args.embed_size
epochs = args.epochs

if args.opt == 'Adam':
    opt = Adam(lr=args.lr)

train_shape = (args.img_height, args.img_width, args.img_channels)

num_layers = args.num_layers
num_units = args.num_units



# Load data
print('Reading concepts...')
concepts_filename = args.concepts_filename
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

print('Loading train data...', flush=True)
train_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv',
    'train_resized',
    0,
    concepts,
    num_labels
)

print('Loading valid data...', flush=True)
valid_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv',
    'train_resized',
    1,
    concepts,
    num_labels
)

print('Loading test data...', flush=True)
test_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv',
    'valid_resized',
    2,
    concepts,
    num_labels
)

infer_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv',
    'test_resized',
    3,
    concepts,
    num_labels
)



# Build and compile model
input_img = Input(train_shape)
feature_extractor = VGG16(weights="imagenet", include_top=False, pooling='avg', input_tensor=Input(shape=train_shape))
features = feature_extractor(input_img)
features = Dense(512)(features)
features = LeakyReLU(0.2)(features)
features = Dropout(0.25)(features)
features = Dense(512)(features)
features = LeakyReLU(0.2)(features)
concept_detectors = []

for i in range(num_layers):
    new_concept_detector = build_classification_model(i, num_units, opt)
    if i == 0:
        output = new_concept_detector(features)
    else:
        output = concatenate([output, new_concept_detector([features, output])])
    concept_detectors.append(new_concept_detector)

make_trainable(feature_extractor, False)
training_model = Model(input_img, output)
training_model.compile(optimizer=opt, loss='binary_crossentropy')
training_model.summary()



# Create path to save the model
path = args.checkpoint_path
if os.path.isdir(path) == True:
    shutil.rmtree(path)
os.makedirs(path)


# Define checkpoint location
checkpoint_2 = ModelCheckpoint(path + '/e_training_weights_{epoch:03d}.h5', period=20) 


# Fit model
training_model.fit_generator(train_data, validation_data=valid_data, epochs=50, steps_per_epoch=len(train_data), workers=4)

# Save model weights
training_model.save_weights(path + '/e_training_weights_final.h5')

# Fit model again with feature extractor trainable
make_trainable(feature_extractor, True)
training_model.compile(optimizer=opt, loss='binary_crossentropy')
training_model.fit_generator(train_data, validation_data=valid_data, callbacks=[checkpoint_2], epochs=100, steps_per_epoch=len(train_data), workers=4)







# Generate results for testin data and show them to the user
for i in tqdm(range(len(test_data))):
    if i == 0:
        test_data_imgs, test_data_lbs = test_data[i]
        pred = training_model.predict(test_data_imgs)
    else:
        test_data_imgs, new_lbs = test_data[i]
        test_data_lbs = np.concatenate((test_data_lbs, new_lbs))
        pred = np.concatenate((pred, training_model.predict(test_data_imgs)))

print(test_data_lbs.shape)
print(pred.shape)
y_pred = np.where(pred >= 0.5, 1, 0)
show_metrics(test_data_lbs, y_pred)



# Generate .CSV to submit to the competition
for i in tqdm(range(len(infer_data))):
    if i == 0:
        test_data_imgs, image_names = infer_data[i]
        pred = training_model.predict(test_data_imgs)
    else:
        test_data_imgs, new_names = infer_data[i]
        image_names = np.concatenate((image_names, new_names))
        pred = np.concatenate((pred, training_model.predict(test_data_imgs)))
y_pred = np.where(pred >= 0.5, 1, 0)

eval_concepts = []
for i in range(len(y_pred)):
    img_concepts = []
    all_ones = np.where(y_pred[i] == 1)[0]
    for idx in all_ones:
        img_concepts.append(dict_concept_inverse[idx])
    eval_concepts.append(';'.join(img_concepts))

eval_set = dict()
eval_set["ID"] = image_names
eval_set["cuis"] = eval_concepts

evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv('./taskA_MmRiM_run20.csv', sep='|', index=False, header=False)
