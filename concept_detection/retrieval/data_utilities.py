# Imports
import numpy as np
import pandas as pd

# TensorFlow Imports
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input



# Class: ImageClefDataset
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
            if self.split:
                self.image_names = self.image_names[:train_ratio]
                self.labels = self.labels[:train_ratio]
            self.shuffle = True

        # VALIDATION
        elif dset == 1 and self.split:
            self.image_names = self.image_names[-valid_ratio:]
            self.labels = self.labels[-valid_ratio:]
            self.offset = train_ratio
        
        print("Number of images: " + str(len(self.image_names)))

        self.on_epoch_end()
    
    def get_all_concepts(self):
        return len(self.concepts)

    def __len__(self):
        return int(np.floor(len(self.image_names) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_labels(self, indexes, chosen_concepts):
        indexes = indexes + self.offset
        itc_matrix = self.image_to_concept_matrix[indexes]
        itc_matrix = itc_matrix[:, chosen_concepts]

        if self.losses == 'all' or self.losses == 'iti':
            m1 = np.repeat(self.image_to_concept_matrix[indexes], len(indexes), axis=0)
            m2 = np.tile(self.image_to_concept_matrix[indexes], (len(indexes), 1))
            vec = m1 + m2
            iou = np.sum(np.where(vec == 2, 1, 0), axis=1) / np.sum(np.where(vec >= 1, 1, 0), axis=1)
            iti_matrix = np.resize(iou, (len(indexes), len(indexes)))
        else:
            iti_matrix = None

        if self.losses == 'all' or self.losses == 'ctc':
            transposed_image_to_concept_matrix = self.image_to_concept_matrix.T[chosen_concepts]
            m1 = np.repeat(transposed_image_to_concept_matrix, len(chosen_concepts), axis=0)
            m2 = np.tile(transposed_image_to_concept_matrix, (len(chosen_concepts), 1))
            vec = m1 + m2
            iou = np.sum(np.where(vec == 2, 1, 0), axis=1) / np.sum(np.where(vec >= 1, 1, 0), axis=1)
            ctc_matrix = np.resize(iou, (len(chosen_concepts), len(chosen_concepts)))
        else:
            ctc_matrix = None
                
        return np.asarray(itc_matrix), np.asarray(iti_matrix), np.asarray(ctc_matrix)

    def get_images(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        names = []
        for i in indexes:
            image = load_img(self.data_folder + '/' + self.image_names[i] + '.jpg',
                grayscale=True,
                target_size=(224, 224),
            )
            image = img_to_array(image)
            image = (image - 127.5) / 127.5
            image = np.reshape(image, (224, 224, 1))
            images.append(image)
            names.append(self.image_names[i])
        
        return np.asarray(images), self.image_to_concept_matrix[indexes], np.asarray(names)

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data = []
        targets_itc = []
        targets_iti = []
        targets_ctc = []
        images = []

        chosen_concepts = range(len(self.concepts))
        if self.dset < 2:
            chosen_concepts = np.random.choice(range(len(self.concepts)), self.batch_size, replace=False)
        
        for i in indexes:
            image = load_img(self.data_folder + '/' + self.image_names[i] + '.jpg',
                grayscale=True,
                target_size=(224, 224),
            )
            image = img_to_array(image)
            image = (image - 127.5) / 127.5
            image = np.reshape(image, (224, 224, 1))
            images.append(image)

        if self.dset < 2 and (self.losses == 'all' or self.losses == 'ctc' or self.losses == 'iti'):
            targets_itc, targets_iti, targets_ctc = self.get_labels(indexes, chosen_concepts)
        else:
            itc_matrix = self.image_to_concept_matrix[indexes + self.offset]
            itc_matrix = itc_matrix[:, chosen_concepts]
    
        if self.dset >= 2:
            return np.asarray(images), self.image_to_concept_matrix[indexes][0]

        if self.losses == 'itc':
            return [np.asarray(images), self.concepts[chosen_concepts]], itc_matrix
        elif self.losses == 'ctc':
            return [np.asarray(images), self.concepts[chosen_concepts]], [targets_itc, targets_ctc]
        elif self.losses == 'iti':
            return [np.asarray(images), self.concepts[chosen_concepts]], [targets_itc, targets_iti]
        return [np.asarray(images), self.concepts[chosen_concepts]], [targets_itc, targets_iti, targets_ctc]



# Class: ImageClefDatasetResNet
class ImageClefDatasetResNet(Sequence):
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
        
        print("Number of images: " + str(len(self.image_names)))

        self.on_epoch_end()
    
    def get_all_concepts(self):
        return len(self.concepts)

    def __len__(self):
        return int(np.floor(len(self.image_names) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_labels(self, indexes, chosen_concepts):
        indexes = indexes + self.offset
        itc_matrix = self.image_to_concept_matrix[indexes]
        itc_matrix = itc_matrix[:, chosen_concepts]

        if self.losses == 'all' or self.losses == 'iti':
            m1 = np.repeat(self.image_to_concept_matrix[indexes], len(indexes), axis=0)
            m2 = np.tile(self.image_to_concept_matrix[indexes], (len(indexes), 1))
            vec = m1 + m2
            iou = np.sum(np.where(vec == 2, 1, 0), axis=1) / np.sum(np.where(vec >= 1, 1, 0), axis=1)
            iti_matrix = np.resize(iou, (len(indexes), len(indexes)))
        else:
            iti_matrix = None

        if self.losses == 'all' or self.losses == 'ctc':
            transposed_image_to_concept_matrix = self.image_to_concept_matrix.T[chosen_concepts]
            m1 = np.repeat(transposed_image_to_concept_matrix, len(chosen_concepts), axis=0)
            m2 = np.tile(transposed_image_to_concept_matrix, (len(chosen_concepts), 1))
            vec = m1 + m2
            iou = np.sum(np.where(vec == 2, 1, 0), axis=1) / np.sum(np.where(vec >= 1, 1, 0), axis=1)
            ctc_matrix = np.resize(iou, (len(chosen_concepts), len(chosen_concepts)))
        else:
            ctc_matrix = None
                
        return np.asarray(itc_matrix), np.asarray(iti_matrix), np.asarray(ctc_matrix)

    def get_images(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        names = []
        for i in indexes:
            image = load_img(self.data_folder + '/' + self.image_names[i] + '.jpg',
                target_size=(224, 224),
            )
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            images.append(image[0])
            names.append(self.image_names[i])
        
        return np.asarray(images), self.image_to_concept_matrix[indexes], np.asarray(names)


    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data = []
        targets_itc = []
        targets_iti = []
        targets_ctc = []
        images = []

        chosen_concepts = range(len(self.concepts))
        if self.dset < 2:
            chosen_concepts = np.random.choice(range(len(self.concepts)), self.batch_size, replace=False)
        
        for i in indexes:
            image = load_img(self.data_folder + '/' + self.image_names[i] + '.jpg',
                target_size=(224, 224),
            )
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            images.append(image[0])

        if self.dset < 2 and (self.losses == 'all' or self.losses == 'ctc' or self.losses == 'iti'):
            targets_itc, targets_iti, targets_ctc = self.get_labels(indexes, chosen_concepts)
        else:
            itc_matrix = self.image_to_concept_matrix[indexes + self.offset]
            itc_matrix = itc_matrix[:, chosen_concepts]
    
        if self.dset >= 2:
            return np.asarray(images), self.image_to_concept_matrix[indexes][0]

        if self.losses == 'itc':
            return [np.asarray(images), self.concepts[chosen_concepts]], itc_matrix
        elif self.losses == 'ctc':
            return [np.asarray(images), self.concepts[chosen_concepts]], [targets_itc, targets_ctc]
        elif self.losses == 'iti':
            return [np.asarray(images), self.concepts[chosen_concepts]], [targets_itc, targets_iti]
        return [np.asarray(images), self.concepts[chosen_concepts]], [targets_itc, targets_iti, targets_ctc]
