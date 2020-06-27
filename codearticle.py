#!/usr/bin/env python
# coding: utf-8




import keras
from keras import backend as K
import matplotlib as mpl
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from os import listdir
from os.path import isfile, join
import numpy as np
import numpy
import cv2
from skimage.io import imread
import skimage
from keras.preprocessing import image
from PIL import Image
import os, sys





mpl.rc('image',cmap='gray')
# Load an color image in grayscale for simplified(not contaminated) solution of electrostatics capacitor and pre processing image
simplified = cv2.imread('simplify.png',cv2.IMREAD_GRAYSCALE)
print(simplified.shape)
plt.imshow(simplified)
plt.show()
#Convert an imagefrom 8-bit to 16-bit signed integer format.
simplified=skimage.img_as_float(simplified, force_copy=False)
simplified = np.array(simplified, dtype=np.float64)
simplified = simplified.reshape(-1, 256,256, 1)
#the scale will be in the range(0,1)
simplified=simplified/np.max(simplified)
plt.imshow(simplified[0,...,0])
plt.show()
print(np.max(simplified))
print(np.min(simplified))
print(simplified.shape)
print(simplified.dtype)
plt.imshow(simplified[0,...,0])


## Load an color images in grayscale for exact(contaminated) solution of electrostatics capacitor and pre-processing images
#550 exact(contaminated) images were generated using simulation and CAD programs
mypath=r'C:\Users\merye\images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
print(len(onlyfiles))
images = numpy.empty(len(onlyfiles),dtype=object)
from skimage import io
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath,onlyfiles[n]),cv2.IMREAD_GRAYSCALE)
    images[n]=cv2.resize(images[n], (256,256))
    plt.imshow(images[n])
    plt.show()
    
    
    images[n] = images[n].reshape((-1, 256,256, 1))
    images[n] = skimage.img_as_float(images[n], force_copy=False)  
    images[n] = np.array(images[n], dtype=np.float64)
    images[n] = images[n]/np.max(images[n])
    print(images[0].shape)


#reshape the matrix (len,256,256,1)
images = np.vstack(images).astype(np.float64)
print(images.shape)
print(images.dtype)
print(len(images))
print(np.max(images[1]))

#taken differences between exact and simplified solution images obtained from Elmer simulation
#with this way, we obtain first element of data pairs to train the network
dif=numpy.empty(len(onlyfiles), dtype=object)
for i in range(0,len(onlyfiles)):
    dif[i]=abs(images[i]-simplified)
    threshold=np.max(dif[i])/4
    dif[i][dif[i] < threshold] = 0
    dif[i][dif[i] > 0] = 1
plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i, ..., 0], cmap='gray')
    plt.gray()
plt.figure()
print(dif.shape)
plt.imshow(dif[1][0,...,0], cmap='gray')


dif = np.vstack(dif).astype(np.float64)
print(dif.dtype)
print(dif.shape)
len(dif)


plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(dif[i,...,0], cmap='gray')
    plt.gray()


# shows differences between exact and simplified geometries
#with this way, we obtain second element of data pairs to train the network
import numpy
import numpy as np
A=6
B=6
W=0.2 # Width of contamination to be read in
H=0.2 # Height of contamination, to be read in
Pc=8.4
Wind = max(round(W/A*255), 1)
Hind = max(round(H/B*255), 1)
dPind = (Pc-1)/10
X_file = open('axisn.txt','r')
X = []
for line in X_file:
    X.extend([float(i) for i in line.split()])
#X=[2.9,2.9,2.801]
#Y=[3.3,5.2,2.7]
Y_file = open('ordinaten.txt','r')
Y = []
for line in Y_file:
    Y.extend([float(i) for i in line.split()])
for i in range(0,len(X)):
    X[i]=max(round(X[i]/A*255),1)
    Y[i]=max(round(Y[i]/B*255),1)

M = numpy.zeros((len(onlyfiles),256,256,1),dtype=float)
print(M.shape)
for s in range(0,len(M)):
    Wnew=X[s]+Wind-1
    Hnew=Y[s]+Hind-1
    if Wnew > 255:
        Wnew = 255
    for c in range(X[s],Wnew):
        if Hnew > 255:
            Hnew = 255
        for r in range(Y[s],Hnew):
            M[s][r][c][0]=dPind
                
    if len(M[s][M[s] != 0]) == 0:
        print(s,":",X[s],Wnew,Y[s],Hnew)
print(M.shape)
print(M[8][M[8] != 0])


print(M.shape)
plt.imshow(M[8,...,0])
print(M.shape)



pip install sklearn


#split data as %80 training and %20 testing
from sklearn.model_selection import train_test_split
train_E,valid_E,train_T,valid_T= train_test_split(M,dif,
                                                  test_size=0.2,random_state=13)

plt.figure(figsize=(20,4))
print("Input")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_E[i, ..., 0], cmap='gray')
    plt.gray()
plt.figure(figsize=(20,4))
print("Output")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_T[i, ..., 0], cmap='gray')
    plt.gray()


print(len(train_E))
print(train_E.shape)
print(train_T.shape)
print(len(valid_E))
print(valid_E.shape)
print(valid_T.shape)

#Hyper parameters
number_epochs = 60
batch_size = 16
learning_rate = 0.0001
num_workers=0
in_channel=1
#for Id in range(0,len(testfiles)):
m,n=256,256
input_images= Input(shape = (m,n,in_channel))

input_images

#Convolutional Autoencoder 
def autoencoder(input_images):

#encoder
    CA1 = Conv2D(256, (3, 3), activation='relu', padding='same')(input_images)
    MP1 = MaxPooling2D((2, 2), padding='same')(CA1)
    CA2 = Conv2D(128, (3, 3), activation='relu', padding='same')(MP1)
    MP2 = MaxPooling2D((2, 2), padding='same')(CA2)
    CA3 = Conv2D(64, (3, 3), activation='relu', padding='same')(MP2)
    MP3 = MaxPooling2D((2, 2), padding='same')(CA3)
    CA4 = Conv2D(32, (3, 3), activation='relu', padding='same')(MP3)
    
    
    encoded = MaxPooling2D((2, 2), padding='same')(CA4)
 #decoder
    CA6 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    UP1 = UpSampling2D((2, 2))(CA6)
    CA7 = Conv2D(64, (3, 3), activation='relu', padding='same')(UP1)
    UP2 = UpSampling2D((2, 2))(CA7)
    CA8 = Conv2D(128, (3, 3), activation='relu', padding='same')(UP2)
    UP3 = UpSampling2D((2, 2))(CA8)
    CA9 = Conv2D(256, (3, 3), activation='relu', padding='same')(UP3)
    UP4 = UpSampling2D((2, 2))(CA9)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(UP4)
    return decoded
#optimization network
autoencoder = Model(input_images, autoencoder(input_images))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')

autoencoder.summary()

#increasing amount of training data
from keras.preprocessing.image import ImageDataGenerator
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,
    width_shift_range=0.2, height_shift_range=0.2,
    vertical_flip=True, fill_mode="nearest")
repeats = 3 # repeats*len(train_E) training set size
aug_E = np.empty((repeats*len(train_E),256,256,1))
aug_T = np.empty((repeats*len(train_T),256,256,1))
autoencoder_train = autoencoder.fit(aug_E, aug_T,batch_size=batch_size,epochs=number_epochs,verbose=1,validation_data=(valid_E, valid_T))

#training and validation loss graph
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(number_epochs)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()

#make a prediction on test images
prediction = autoencoder.predict(M)


plt.figure(figsize=(20,4))
print("Test Images")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(M[i, ..., 0], cmap='gray')
    plt.gray()
    
plt.figure(figsize=(20,4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(prediction[i, ..., 0], cmap='gray')
    plt.gray()
plt.show()


preds_0 = prediction[0]*255
preds_0 = preds_0.reshape(256,256)
x_test_0 = M[0]*255
x_test_0 = x_test_0.reshape(256,256)
plt.imshow(x_test_0, cmap='gray')
plt.gray()



plt.imshow(preds_0, cmap='gray')
plt.gray()


