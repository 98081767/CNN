# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:57:33 2020

@author: archel

@code copied from: https://www.kaggle.com/rajmehra03/flower-recognition-cnn-keras
as an example of how to use CNN


Context

This dataset contains 4242 images of flowers.
The data collection is based on the data flicr, google images, yandex images.
You can use this datastet to recognize plants from the photo.


Content

The pictures are divided into five classes: chamomile, tulip, rose, sunflower, dandelion.
For each class there are about 800 photos. Photos are not high resolution, about 320x240 pixels. Photos are not reduced to a single size, they have different proportions!


"""

#import libraries
import os

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#%matplotlib inline  
#style.use('fivethirtyeight')
#sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image


#set current working directory
os.chdir('C:/Users/arche/Documents/UTS/Python-References/94691 Deep Learning/CNN')

print(os.getcwd())
print(os.listdir('./flowers-recognition/flowers'))

#global variables
X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='./flowers-recognition/flowers/daisy'
FLOWER_SUNFLOWER_DIR='./flowers-recognition/flowers/sunflower'
FLOWER_TULIP_DIR='./flowers-recognition/flowers/tulip'
FLOWER_DANDI_DIR='./flowers-recognition/flowers/dandelion'
FLOWER_ROSE_DIR='./flowers-recognition/flowers/rose'

os.listdir(FLOWER_ROSE_DIR)

#global functions

def assign_label(img,flower_type):
    return flower_type
    
def make_train_data(flower_type,DIR):
    
    #tqdm() show progress for loops - awesome
    for img in tqdm(os.listdir(DIR)):
        label   = assign_label(img,flower_type)
        path    = os.path.join(DIR,img)
        #print(path)
        img     = cv2.imread(path,cv2.IMREAD_COLOR)
        img     = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
    
make_train_data('Daisy',FLOWER_DAISY_DIR)
#print(len(X))
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
#print(len(X))
make_train_data('Tulip',FLOWER_TULIP_DIR)
#print(len(X))
make_train_data('Dandelion',FLOWER_DANDI_DIR)
#print(len(X))
make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))

#show some random images
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)

for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()


#use label encoder to perform one hot encoding
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)

#X is current a list. Convert to a numpy array
X=np.array(X)
#X.shape (4323, 150, 150, 3)
#4323 = number of images
#150 x 150 = size of image
#3 = RGB channels

#>> X[0].shape - gets the first image.

#divide RGB images by 255.
#why: 
#   https://stackoverflow.com/questions/20486700/why-we-always-divide-rgb-values-by-255
#   RGB (Red, Green, Blue) are 8 bit each.
#   The range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
#   The combination range is 256*256*256.
#   By dividing by 255, the 0-255 range can be described with a 0.0-1.0 range where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF).
X=X/255

#split into training and validation sets
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

#set random seeds
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


#modelling

# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))
