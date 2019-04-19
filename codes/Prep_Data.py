import numpy as np
#import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
from Constant import *
import pickle
import gc
import keras.applications.vgg16
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Activation


vgg16 = keras.applications.vgg16.VGG16()
vgg16.layers.pop()
vgg16.layers.pop()
vgg16.layers.pop()
vgg16.layers.pop()
vgg16.layers.pop()


model = Sequential()
for layer in vgg16.layers:
    model.add(layer)


model.add(BatchNormalization())


model.summary()

file = open(DATASET_INDEX)

INDEX = np.loadtxt(file, delimiter=',', dtype='str')

"""
    This method creates the numpy training data

    from the images dataset, for each video individualy

    """


def create_training_data():
    print('Converting Training video images to numpy array ...')
    for index in INDEX:
        path_train = TR_IMG_DIR + index + '/images/'
        #path_train = os.path.join(TR_IMG_DIR1, index)
        training_data = []

        for img in tqdm(os.listdir(path_train)):
            img_array = cv2.imread(os.path.join(path_train, img) ,cv2.COLOR_RGB2BGR)
            new_array = cv2.resize(img_array, IMG_SIZE).astype(float)
            #im = np.expand_dims(new_array, axis=0)
            #a = model.predict(im)

            training_data.append([new_array])

        training_data = np.array(training_data).reshape(-1, 224, 224, 3)
        np.save(TR_VID_DIR + index, training_data)

    del training_data
    gc.collect()
    print('Converting Training video images to numpy array ok ...')


"""
    This method creates the numpy Ground truth data0

    from the images dataset, for each video individualy

    """


def create_GT_data():
    print('Converting Ground truth video images to numpy array ...')
    j = 0
    for index in INDEX:
        print('indexe'+ str(index))
        path_train = GT_IMG_DIR + index + '/maps/'
        #path_train = os.path.join(GT_IMG_DIR1, index)
        GT_data = []

        for img in tqdm(os.listdir(path_train)):

            img_array = cv2.imread(os.path.join(path_train, img))
            new_array = cv2.resize(img_array, IMG_SIZE)
            gray_image = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
            gray_image = gray_image.astype(float)
            im = gray_image/255.0
            GT_data.append([im])

        Y = []

        for features in GT_data:
            Y.append(features)

        Y = np.array(Y).reshape(-1, 224, 224, 1)
        filename = os.path.join(GT_VID_DIR, index)
        np.save(filename, Y)
    del Y
    del GT_data
    gc.collect()
    print('Converting Ground truth video images to numpy array ok...')


"""
    This method creates the batches training data, by applying a sliding window on each 3 succesive frames

    from the images dataset, for each video individualy

    """

def generator_creation():
    print('Batch Creation function ...')
    o = '.npy'
    s = 'G:/WORKSPACE/DATA/GT_DATA1/'
    f = open("eric.txt", "w+")
    for index in INDEX:

        path_train = os.path.join(s, index + o)
        X = np.load(path_train)

        print('indexe : ' + str(index) +'---'+ str(len(X)))
        for j in range(len(X)):
            f.write("'"+index+'__'+str(j)+"',")
            f.write("\r\n")
    f.close()

def feature_map_function():
    print('feature map creation with VGG-16 ...')
    o = '.npy'
    j = 0
    for index in INDEX:

        path_train = os.path.join(TR_VID_DIR, index + o)
        X = np.load(path_train)
        n = []
        k = []
        m = []
 
        print(len(X))

        for i in range(len(X)):
            n = model.predict(X[i:i+1,:,:,:])
            k = n[0]
            m.append(k)
        np.save(TR_VGG_DIR +index, m)
        print(str(j))
        j = j+1


def Batch_Creation():
    print('Batch Creation function ...')
    o = '.npy'
    j = 0
    for index in INDEX:

        path_train = os.path.join(TR_VGG_DIR, index + o)
        X = np.load(path_train)
        S = []
        K = []
        for i in range(len(X)):
            if   i==0:
                K = np.array([X[i], X[i], X[i], X[i], X[i], X[i]])
            elif i==1:
                K = np.array([X[i-1], X[i-1], X[i], X[i], X[i], X[i]])
            elif i==2:
                K = np.array([X[i-2], X[i-1], X[i-1], X[i], X[i], X[i]])
            elif i==3:
                K = np.array([X[i-3], X[i-2], X[i-1], X[i], X[i], X[i]])
            elif i==4:
                K = np.array([X[i-4], X[i-3], X[i-2], X[i-1], X[i], X[i]])
            elif i==5:
                K = np.array([X[i-5], X[i-4], X[i-3], X[i-2], X[i-1], X[i]])
            else:
                K = np.array([0.4*X[i-5], 0.5*X[i - 4], 0.6*X[i - 3], 0.7*X[i - 2], 0.8*X[i - 1], X[i]])

            S.append(K)
            np.save( TR_BATCH_DIR+index + '__' + str(i),K)

        np.save(TR_BATCH_DIR+index, S)


        """""
	if you choose 3 frames
        for i in range(len(X)):
            if i == 0:
                K = np.array([X[i], X[i], X[i]])
            elif i == 1:
                K = np.array([X[i], X[i], X[i-1]])
            elif i == 2:
                K = np.array([X[i], X[i - 1], X[i-2]])
            else:
                K = np.array([X[i], X[i - 1], X[i-2]])
            np.save(TR_BATCH_DIR + index+'____'+str(i), K)
            print(str(j))
            j= j+1
            #S.append(K)
        """""


    #del S
    gc.collect()


    print('Batch Creation function ok...')

"""
    #This method adds a 5 dimension to the numpy Ground truth data, so it could be compatible with a 3d conv

    """

def Batch_GT_Creation():
    o = '.npy'

    for index in INDEX:
        path_train = os.path.join(GT_VID_DIR, index + o)
        Y = np.load(path_train)

        for i in range(len(Y)):
            K = np.array([Y[i]])
            np.save(GT_BATCH_DIR + index+'__'+str(i), K)
    gc.collect()


"""
    This method creates a gloabl data array by concatenating each numpy array video

    """


def Global_Train_Data():
    print('Global Train Data function ...')
    o = '.npy'
    p = np.zeros([1, 12, 224, 224, 3])
    i = 0
    for index in INDEX:
        path_train = os.path.join(TR_BATCH_DIR, index + o)
        L = np.load(path_train)
        p = np.append(p, L, axis=0)
        print(i)
        i = i+1
    np.save(TR_GLOBAL_DIR, p)
    del L
    del p
    gc.collect()
    print('Global Train Data function ...')


"""
    This method creates a gloabl data array by concatenating each numpy array video

    """


def Global_GT_Data():
    j = '.npy'
    n = np.zeros([1, 1, 224, 224, 1])
    for index in INDEX:
        path_train = os.path.join(GT_BATCH_DIR, index + j)
        a = np.load(path_train)
        n = np.append(n, a, axis=0)


    np.save(GT_GLOBAL_DIR, n)
    del a

    gc.collect()
def Create_Data_Set():
    print('PreProcessing Dataset =========================================')

    print('Processing Training Data =================================================')

    create_training_data()
    gc.collect()
    feature_map_function()
    Batch_Creation()
    print('Processing Ground truth data =============================================')
    create_GT_data()
    Batch_GT_Creation()
    gc.collect()



Create_Data_Set()
