import tensorflow as tf
import sys
import os
import keras
from datetime import datetime
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Activation

def cnn_network():
    model = Sequential()
    """""
    # Layer 1
    model.add(
        Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), input_shape=(6, 224, 224, 3), use_bias=64, padding='SAME',
               activation='relu',
               name='conv3D1_1'))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=64, padding='SAME', activation='relu',
                     name='conv3D1_2'))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=64, padding='SAME', activation='relu',
                     name='conv3D1_29'))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(BatchNormalization())

    # Layer 2

    model.add(Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=128, padding='SAME', activation='relu',
                     name='conv3D2_1'))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=128, padding='SAME', activation='relu',
                     name='conv3D2_29'))


    model.add(Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=128, padding='SAME', activation='relu',
                     name='conv3D2_2'))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(BatchNormalization())

    # Layer 3

    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=256, padding='SAME',
                     activation='relu', name='conv3D_5'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=256,
                     padding='SAME', activation='relu', name='conv3D_6'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(BatchNormalization())

    # Layer 4

    model.add(Conv3D(256, kernel_size=(2, 3, 3), strides=(1, 1, 1), use_bias=256,
                     padding='SAME', activation='relu', name='conv3D_7'))

    model.add(Conv3D(256, kernel_size=(2, 3, 3), strides=(1, 1, 1),
                     use_bias=256, padding='SAME', activation='relu', name='conv3D_8'))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(BatchNormalization())
    """

    # Layer 4
    model.add(
    Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), input_shape=(6, 14, 14, 512), use_bias=512, padding='SAME',
               activation='relu',
               name='conv3D_1_1'))

    model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1),use_bias=512, padding='SAME',
           activation='relu',name='conv3D_1_5'))

    model.add(MaxPooling3D(pool_size=(4, 2, 2), strides=(1, 2, 2)))
    model.add(BatchNormalization())
    """""
    model.add(
        Conv3D(1024, kernel_size=(2, 3, 3), strides=(1, 1, 1), use_bias=1024,
               padding='SAME',
               activation='relu',
               name='conv3D_1_12'))
    
    #model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=512, padding='SAME',
    #                 activation='relu', name='conv3D_1_59'))

    model.add(MaxPooling3D(pool_size=(3, 1, 1), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    
    """""
    #model.add(Conv3D(1024, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=1024, padding='SAME', activation='relu',
                     #name='conv3D_1_2'))

    #model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=512, padding='SAME', activation='relu',
                     #name='conv3D_10'))

    #model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1)))

    #model.add(BatchNormalization())
    # Layer 5


    #model.add(BatchNormalization())
    # Layer 2
    model.add(
        Conv3DTranspose(512, kernel_size=(1, 3, 3), strides=(1, 2, 2), use_bias=512, padding='SAME', activation='relu',
                        name='Deconv3D_16'))

    model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=512,
                     padding='SAME', activation='relu', name='conv3D_2__19'))

    #model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=512,
     #                padding='SAME', activation='relu', name='conv3D_2__77'))

    model.add(
    Conv3DTranspose(256, kernel_size=(3, 3, 3), strides=(1, 2, 2), use_bias=256, padding='SAME', activation='relu',
                        name='Deconv3D_1'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=256,
                     padding='SAME', activation='relu', name='conv3D_2__1'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bias=256, padding='SAME', activation='relu',
                     name='conv3D_2__2'))
    model.add(MaxPooling3D(pool_size=(3, 1, 1), strides=(1, 1, 1)))
    #model.add(Conv3D(256, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=256, padding='SAME', activation='relu',
                     #name='conv3D_18'))
    #model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1)))
    model.add(BatchNormalization())
    # Layer 3

    model.add(
        Conv3DTranspose(128, kernel_size=(1, 3, 3), strides=(1, 2, 2), use_bias=128, padding='SAME', activation='relu',
                        name='Deconv3D_2'))

    model.add(Conv3D(128, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=128, padding='SAME', activation='relu',
                     name='conv3D__3__1'))

    model.add(Conv3D(128, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=128, padding='SAME', activation='relu',
                     name='conv3D__3__2'))
    model.add(BatchNormalization())

    # Layer4

    model.add(
        Conv3DTranspose(64, kernel_size=(1, 3, 3), strides=(1, 2, 2), use_bias=64, padding='SAME', activation='relu',
                        name='Deconv3D_3'))

    model.add(Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=64, padding='SAME', activation='relu',
                     name='conv3D__4__1'))

    model.add(Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=64, padding='SAME', activation='relu',
                         name='conv3D__4__2'))


    model.add(BatchNormalization())
    # Layer 5

    model.add(
        Conv3DTranspose(32, kernel_size=(1, 3, 3), strides=(1, 2, 2), use_bias=32, padding='SAME', activation='relu',
                        name='Deconv3D_4'))

    model.add(Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=32, padding='SAME', activation='relu',
                     name='conv3D__5__1'))

    model.add(Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=16, padding='SAME', activation='relu',
                     name='conv3D__5__2'))

    #model.add(Conv3D(8, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=8, padding='SAME', activation='relu',
                     #name='conv3D2_23'))

    model.add(Conv3D(1, kernel_size=(1, 3, 3), strides=(1, 1, 1), use_bias=1, padding='SAME', activation='sigmoid',
                                         name='conv3D__5__3'))
    #model.add(BatchNormalization())

    return model

k = cnn_network()
k.summary()