# Imports
import os
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm

# Define Keras Backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Tensorflow & Keras Imports
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

# Sklearn Imports
import sklearn.metrics

# Project Imports
from load_autoregressive_data import ImageClefDataset



# Training hyperparameters
num_labels = 256
embed_size = 2048
epochs = 1000
opt = Adam(lr=1e-5)
train_shape = (224, 224, 3)
num_layers = 17
num_units = 125



# Load data
print('Reading concepts...')
concepts_filename = 'top2125_concepts.csv'
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
    'train_resized', 0,
    concepts, num_labels
)

print('Loading valid data...', flush=True)
valid_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv',
    'train_resized', 1,
    concepts, num_labels
)

print('Loading test data...', flush=True)
test_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv',
    'valid_resized', 2,
    concepts, num_labels
)

infer_data = ImageClefDataset(
    dict_concept,
    'ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv',
    'test_resized', 3,
    concepts, num_labels
)



# Function: Make model trainable
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
      l.trainable = val



# Function: Build classification model
def build_classification_model(iteration):
    feat_input = Input((512,))
    if iteration == 0:
        x = feat_input
    else:
        label_input = Input((iteration * num_units,))
        x = concatenate([feat_input, label_input])

    x = Dense(num_units, activation='sigmoid')(x)
    if i == 0:
        label_encoder = Model(feat_input, x)
    else:
        label_encoder = Model([feat_input, label_input], x)
    label_encoder.compile(loss='binary_crossentropy', optimizer=opt)
    return label_encoder



# Build and compile model
input_img = Input(train_shape)

feature_extractor = VGG16(weights="imagenet", include_top=False, pooling='avg', input_tensor=Input(shape=(224, 224, 3)))

features = feature_extractor(input_img)
features = Dense(512)(features)
features = LeakyReLU(0.2)(features)
features = Dropout(0.25)(features)
features = Dense(512)(features)
features = LeakyReLU(0.2)(features)
concept_detectors = []
for i in range(num_layers):
    new_concept_detector = build_classification_model(i)
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
path = "./models_vgg"
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



# Function: Show metrics
def show_metrics(y_true, y_pred):
    print('Exact Match Ratio: ' + str(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: ' + str(sklearn.metrics.hamming_loss(y_true, y_pred)))
    print('Recall: ' + str(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: ' + str(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1-Score: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))



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
