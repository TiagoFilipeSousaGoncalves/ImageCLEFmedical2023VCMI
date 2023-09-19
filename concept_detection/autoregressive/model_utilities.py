# Imports
import os

# Tensorflow & Keras Imports
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model

# Sklearn Imports
import sklearn.metrics



# Function: Make model trainable
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
      l.trainable = val



# Function: Build classification model
def build_classification_model(iteration, num_units, opt):
    feat_input = Input((512,))
    if iteration == 0:
        x = feat_input
    else:
        label_input = Input((iteration * num_units,))
        x = concatenate([feat_input, label_input])

    x = Dense(num_units, activation='sigmoid')(x)
    if iteration == 0:
        label_encoder = Model(feat_input, x)
    else:
        label_encoder = Model([feat_input, label_input], x)
    label_encoder.compile(loss='binary_crossentropy', optimizer=opt)
    return label_encoder



# Function: Show metrics
def show_metrics(y_true, y_pred):
    print('Exact Match Ratio: ' + str(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: ' + str(sklearn.metrics.hamming_loss(y_true, y_pred)))
    print('Recall: ' + str(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: ' + str(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1-Score: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))
