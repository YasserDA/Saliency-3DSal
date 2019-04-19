import numpy as np
import keras
from  keras import backend as K
from Constant import *

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=32,
                 shuffle=False):

        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 6, 14, 14, 512))
        y = np.empty((self.batch_size, 1, 224, 224, 1))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(TR_BATCH_DIR + ID + '.npy')
            # Store ground truth
            y[i,] = np.load(GT_BATCH_DIR + ID + '.npy')

        return X, y
