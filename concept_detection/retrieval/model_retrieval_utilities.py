# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# TensorFlow Imports
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Model

# Sklearn Imports
import sklearn.metrics



# Function: Build the feature extractor
def build_feature_extractor(train_shape, embed_size):
    img_input = Input(train_shape)
    h1 = Conv2D(int(embed_size / 8), (5, 5), activation = 'relu', padding = 'same', name = 'id_conv1')(img_input)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    h1 = Conv2D(int(embed_size / 4), (5, 5), activation = 'relu', padding = 'same', name = 'id_conv2')(h1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    h1 = Conv2D(int(embed_size / 2), (3, 3), activation = 'relu', padding = 'same', name = 'id_conv3')(h1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    h1 = Conv2D(embed_size, (3, 3), activation = 'relu', padding = 'same', name = 'id_conv4')(h1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    features = GlobalAveragePooling2D()(h1)
    features = Dense(embed_size, name = 'medical_features')(features) # activation = 'tanh',

    feat_extractor = Model(img_input, features)

    return feat_extractor



# Function: Build label encoder
def build_label_encoder(all_concepts, embed_size):
    label_input = Input((len(all_concepts),))
    x = Dense(embed_size / 2)(label_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(embed_size)(x)
    label_encoder = Model(label_input, x)
    return label_encoder



# Function: Make model layers trainable
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
      l.trainable = val



# Function: Build contrastive loss
def contrastive_loss(y_true, y_pred, margin=1):
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



# Function: Build distance function
def distance_lambda(image_encodings, label_encodings):
    A_dots = K.sum(image_encodings*image_encodings, axis=1, keepdims=True)*np.ones(shape=(1,num_labels))
    B_dots = K.sum(label_encodings*label_encodings, axis=1)*np.ones(shape=(num_labels,1))
    D_squared = A_dots + B_dots -2*K.dot(image_encodings, K.transpose(label_encodings))
    return K.sqrt(D_squared)



# Function: Euclidean Distance (NumPy)
def np_euclidean_distance(image_encodings, label_encodings):
    dist = []
    for lab in label_encodings:
      dist.append(np.sqrt(np.sum(np.square(image_encodings[0] - lab))))
    return np.asarray(dist)



# Function: Get predictions
def get_pred(norm_matrix, threshold, top1_predicted_labels):
    y_pred = np.where(norm_matrix < threshold, 1, 0)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
    for i in range(len(y_pred)):
        y_pred[i][np.where(top1_predicted_labels[i] == 1)[0]] = 1
    return y_pred



# Function: Show metrics
def show_metrics(y_true, y_pred):
    print('Exact Match Ratio: ' + str(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: ' + str(sklearn.metrics.hamming_loss(y_true, y_pred)))
    print('Recall: ' + str(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: ' + str(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1-Score: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))



# Function: Get threshold
def get_threshold(dataset, feature_extractor, encoded_labels, all_concepts):
    euclidean_distance_matrix = []
    y_true = []
    top1_predicted_labels = []
    for idx in tqdm(range(0, len(dataset))):    
        img_data, lbl_data, name_data = dataset.get_images(idx)
        image_features = feature_extractor.predict(img_data)
        for i in range(len(img_data)):
            y_true.append(lbl_data[i])
            distance_image = np_euclidean_distance(image_features[i], encoded_labels)
            euclidean_distance_matrix.append(distance_image)
            top1 = distance_image.argsort()[0]
            pred = np.zeros((len(all_concepts),))
            pred[top1] = 1
            top1_predicted_labels.append(pred)
    y_true = np.asarray(y_true)
    top1_predicted_labels = np.asarray(top1_predicted_labels)

    norm_matrix = (euclidean_distance_matrix-np.amin(euclidean_distance_matrix))/(np.amax(euclidean_distance_matrix)-np.amin(euclidean_distance_matrix))
    matrix_to_array = np.reshape(norm_matrix, (norm_matrix.shape[0]*norm_matrix.shape[1],))
    ytrue_squeeze = np.reshape(y_true, (np.asarray(y_true).shape[0]*np.asarray(y_true).shape[1],))
    absent_concepts = matrix_to_array[np.where(ytrue_squeeze == 0)[0]]
    present_concepts = matrix_to_array[np.where(ytrue_squeeze == 1)[0]]
    print('Average distance in existing concepts: ' + str(np.average(present_concepts)))
    print('STD distance in existing concepts: ' + str(np.std(present_concepts)))
    print('Average distance in absent concepts: ' + str(np.average(absent_concepts)))
    print('Average distance overall: ' + str(np.average(matrix_to_array)))
    perc_absent = len(np.where(absent_concepts < np.average(present_concepts))[0]) / len(absent_concepts)
    perc_absent_std = len(np.where(absent_concepts < np.average(present_concepts) - np.std(present_concepts))[0]) / len(absent_concepts)
    perc_exist = len(np.where(present_concepts > np.average(absent_concepts))[0]) / len(present_concepts)
    print('Percentage of existing concepts > avg dist absent: ' + str(perc_exist))
    print('Percentage of absent concepts < avg dist existing: ' + str(perc_absent))
    print('Percentage of absent concepts < avg - std dist existing: ' + str(perc_absent_std))
    
    avg = np.average(present_concepts)
    std = np.std(present_concepts)

    threshold_std = avg - 0.5 * std

    print('\nResults on validation: ' + str(threshold_std))
    y_pred = get_pred(norm_matrix, threshold_std, top1_predicted_labels)
    show_metrics(y_true, y_pred)
    
    return threshold_std



# Function: Inference, from threshold
def infer(threshold, test_data, feature_extractor, encoded_labels, all_concepts, dict_concept_inverse):
    euclidean_distance_matrix = []
    top1_predicted_labels = []
    eval_concepts = []
    eval_images = []
    
    for idx in tqdm(range(0, len(test_data))):
        eval_images.append(test_data.image_names[idx])
        infer_data_img, _ = test_data[idx]
        image_features = feature_extractor.predict(infer_data_img)
        distance_image = np_euclidean_distance(image_features, encoded_labels)
        euclidean_distance_matrix.append(distance_image)
        top1 = distance_image.argsort()[0]
        pred = np.zeros((len(all_concepts),))
        pred[top1] = 1
        top1_predicted_labels.append(pred)

    norm_matrix = (euclidean_distance_matrix-np.amin(euclidean_distance_matrix))/(np.amax(euclidean_distance_matrix)-np.amin(euclidean_distance_matrix))

    y_pred = np.where(norm_matrix < threshold, 1, 0)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
    for i in range(len(y_pred)):
        img_concepts = []
        y_pred[i][np.where(top1_predicted_labels[i] == 1)[0]] = 1
        all_ones = np.where(y_pred[i] == 1)[0]
        for idx in all_ones:
            img_concepts.append(dict_concept_inverse[idx])
        eval_concepts.append(';'.join(img_concepts))
    eval_set = dict()
    eval_set["ID"] = eval_images
    eval_set["cuis"] = eval_concepts

    evaluation_df = pd.DataFrame(data=eval_set)
    evaluation_df.to_csv('./retrieval_results.csv', sep='|', index=False, header=False)
