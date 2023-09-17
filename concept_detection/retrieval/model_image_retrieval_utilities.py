# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Sklearn Imports
import sklearn.metrics



# Function: Euclidean Distance (NumPy)
def np_euclidean_distance(image_encodings, label_encodings):
    dist = []
    for lab in label_encodings:
        dist.append(np.sqrt(np.sum(np.square(image_encodings - lab))))
    return np.asarray(dist)



# Function: Show metrics
def show_metrics(y_true, y_pred):
    print('Exact Match Ratio: ' + str(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: ' + str(sklearn.metrics.hamming_loss(y_true, y_pred)))
    print('Recall: ' + str(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: ' + str(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1-Score: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))



# Function: Retrieve concepts
def retrieve(dataset, top_images, train_data, model, method=1):
    y_true = []

    for idx in tqdm(range(0, len(train_data))):  
        train_img, train_lbl, img_names = train_data.get_images(idx)
        if idx == 0:
            train_features = model.predict(train_img)
            train_labels = train_lbl
            train_names = img_names
        else:
            train_features = np.concatenate((train_features, model.predict(train_img)))
            train_labels = np.concatenate((train_labels, train_lbl))
            train_names = np.concatenate((train_names, img_names))

    print(train_features.shape)
    print(train_labels.shape)
    print(train_names.shape)

    closest_image_names = []
    y_pred = []
    image_names = []
    for idx in tqdm(range(0, len(dataset))):  
        img_data, lbl_data, name_data = dataset.get_images(idx)
        image_features = model.predict(img_data)
        for i in range(len(img_data)):
            image_names.append(name_data[i])  
            y_true.append(lbl_data[i])
            distance_image = np_euclidean_distance(image_features[i], train_features)
            closest_imgs = distance_image.argsort()[:top_images]
            pred_labels = train_labels[closest_imgs[0]]
            if method == 1:
                for c in range(1, top_images):
                    if c == 1:
                        pred_labels = np.logical_and(train_labels[closest_imgs[0]], train_labels[closest_imgs[c]])
                        y_pred.append(pred_labels.astype(int))
                    else:
                        aux_labels = np.logical_and(train_labels[closest_imgs[0]], train_labels[closest_imgs[c]])
                        pred_labels = np.logical_or(pred_labels, aux_labels)
                if len(np.where(pred_labels.astype(int) == 1)[0]) == 0:
                    pred_labels = train_labels[closest_imgs[0]]
                y_pred.append(pred_labels.astype(int))
            if method == 2:
                logic_ands = []
                for c in range(0, top_images):
                    for c2 in range(0, top_images):
                        if c == c2:
                            continue
                        logic_ands.append(train_labels[closest_imgs[c]] * train_labels[closest_imgs[c2]])
                y_pred.append(np.where(np.sum(logic_ands, axis=0) > 0, 1, 0))

    show_metrics(y_true, y_pred)

    eval_set = dict()
    eval_set["ID"] = image_names
    eval_set["Closest"] = closest_image_names

    evaluation_df = pd.DataFrame(data=eval_set)
    evaluation_df.to_csv('./image_retrieval.csv', sep='\t', index=False)
