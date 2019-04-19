from Constant import *
from Prep_Data import *
from Model import *
import keras
import numpy as np
from Gen_class import DataGenerator
from keras.models import Sequential
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from dic import param
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import math
from dict import partition
from keras.models import load_model
import keras.callbacks

#gpu=0
# to using a specific GPU device
#import os
#if gpu == 1:
#   os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#  os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Preparing the dataset
Create_Data_Set()

params = {'batch_size':64,
          'shuffle': False}


# to save memory usage, we use the generator to feed data to 3D CNN
training_gen = DataGenerator(pa['train'], **params)
val_gen = DataGenerator(pa['val'], **params)

print('Data and labels loaded Successfully')


## update the ADAM opmtimize learning rate each 2 epochs
def step_decay(epoch):
    lrate = 0.0
    initial_lrate = 0.0001
    drop = 0.1
    epoch_drop = 2
    lrate = initial_lrate* math.pow(drop, math.floor((epoch)/epoch_drop))

    print(str(lrate))
    return lrate



myPath = os.path.join(Model_DIR, 'model-BCE-COEF-davis-dhf1k-ufc-24.hdf5')

# Instance the 3D CNN architecture
Saliency_model = cnn_network()

opt = keras.optimizers.Adam(lr=0.0, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

Saliency_model.compile(optimizer=opt, loss= 'binary_crossentropy', metrics=['acc'])

Saliency_model.summary()

chekpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=2)

lrate = LearningRateScheduler(step_decay)

callback_list = [chekpoint, lrate]


history = Saliency_model.fit_generator(generator=training_gen, validation_data=val_gen, epochs=33, callbacks= callback_list, verbose=1)

print(history.history.keys())

plt.subplot(221)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train','test'], loc ='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'], loc ='upper left')
plt.show()
