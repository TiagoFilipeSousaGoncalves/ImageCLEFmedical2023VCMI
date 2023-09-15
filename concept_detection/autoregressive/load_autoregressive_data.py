# Imports
import os
import pandas as pd
import numpy as np

# Tensorflow & Keras Imports
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input



# class: ImageClefDataset
class ImageClefDataset(Sequence):
    # dset: train (0), valid (1), test (2), inference (3)
    def __init__(self, dict_concept, data_filename, data_folder, dset, concepts, batch_size, losses='itc', split=True):
        data_csv = pd.read_csv(data_filename, header=0, sep="\t")
        all_image_names = data_csv[:]['ID']
        all_labels = np.array(data_csv[:]['cuis'])
        self.concepts = concepts
        self.batch_size = batch_size
        self.dset = dset
        self.data_folder = data_folder
        self.dict_concept = dict_concept
        self.losses = losses

        self.image_to_concept_matrix = np.zeros((len(all_image_names), len(concepts)))
        for i in range(len(all_image_names)):
            concepts_per_image = all_labels[i].split(";")
            for c in concepts_per_image:
                if c in self.dict_concept:
                    self.image_to_concept_matrix[i][self.dict_concept[c]] = 1

        train_ratio = int(0.95 * len(data_csv))
        valid_ratio = len(data_csv) - train_ratio
        self.image_names = list(all_image_names)
        self.labels = all_labels
        self.shuffle = False
        self.offset = 0

        # TRAIN
        if dset == 0:
            if split:
                self.image_names = self.image_names[:train_ratio]
                self.labels = self.labels[:train_ratio]
            self.shuffle = True

        # VALIDATION
        elif dset == 1 and split:
            self.image_names = self.image_names[-valid_ratio:]
            self.labels = self.labels[-valid_ratio:]
            self.offset = train_ratio
                
        elif dset == 3:
            self.image_names = os.listdir(data_folder)
            self.image_names = [x.split('.')[0] for x in self.image_names]
        
        print("Number of images: " + str(len(self.image_names)))
        self.on_epoch_end()
    
    def get_all_concepts(self):
        return len(self.concepts)

    def __len__(self):
        return int(np.ceil(len(self.image_names) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        if (index+1) * self.batch_size > len(self.image_names):
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        images = []
        names = []

        chosen_concepts = range(len(self.concepts))
        if self.dset < 2:
            chosen_concepts = np.random.choice(range(len(self.concepts)), self.batch_size, replace=False)
        
        for i in indexes:
            image = load_img(self.data_folder + '/' + self.image_names[i] + '.jpg',
                target_size=(224, 224),
                grayscale=False
            )
            image = img_to_array(image)
            # image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            images.append(image)
            names.append(self.image_names[i])
    
        if self.dset == 3:
            return np.asarray(images), np.asarray(names)
        return np.asarray(images), self.image_to_concept_matrix[indexes]
