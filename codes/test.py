#from Constant import *
import os
import keras
import numpy as np
from numpy import *
import tensorflow as tf
import h5py
#import pickle
from keras.models import load_model
from keras.utils import CustomObjectScope
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.misc
from Constant import *



print('Loading Ground truth Data ================================================')
Gt = "/home/yasser/Desktop/3dbear.npy"
X = np.load(Gt)




3DSal-weighted = os.path.join(Model_DIR, '3DSal-weighted.hdf5')
3DSal-Base = os.path.join(Model_DIR, '3DSal-Base.hdf5')


model = load_model(3DSal-weighted)
o = '.npy'

file = open(TEST_INDEX)

INDEX = np.loadtxt(file, delimiter=',', dtype='str')

for index in INDEX:

    path_train = os.path.join(TEST_DIR, index + o)
    X = np.load(path_train)
    j = 0
    for i in range(0,len(X),1):
        Predictions = model.predict(X[i:i+1,:,:,:,:])
        p = Predictions[0][0]
        s = np.array(p[:,:,0])
        scipy.misc.imsave(TEST_RES+index+'/'+str(format(j,'04'))+'.jpg', s)
        #p = p.astype(np.uint8)
        #print(p.shape, p.max(), p.min())
        #print(s.shape, s.max(), s.min())
        #print('------------------------------------------------')
        #im = Image.fromarray(p,'RGB')
        #j = i+1
        #scipy.misc.imsave('G:/TEST-3D-CNN/'+index+'/'+str(format(j,'04'))+'.jpg', s)
        #print(Predictions.shape, Predictions.max(), Predictions.min(),Predictions.mean())
        #cv2.imwrite('frame-'+str(i)+'.jpg', p)
        #cv2.imshow("image", p);
        #cv2.waitKey();
        #plt.imshow(p.reshape(224,224), cmap="gray")
        #plt.imshow(Y[i][0].reshape(224,224), cmap="gray")
        #plt.show()
