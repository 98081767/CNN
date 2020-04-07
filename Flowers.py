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
tf.random.set_seed(42)


#modelling

# # modelling starts using a CNN.
#NOTES:
# - Max Pooling is a way to introduce noise by downsampling to prevent overfitting - See https://protect-au.mimecast.com/s/JqZhCOMK2OTAAEOPiv-kbX?domain=computersciencewiki.org
# - Padding:
#       VALID Padding: it means no padding and it assumes that all the dimensions are valid so that the input image gets fully covered by a filter and the stride specified by you. and its like saying, we are ready to loose some information. 
#       SAME Padding: it applies padding to the input image so that the input image gets fully covered by the filter and specified stride.It is called SAME because, for stride 1 , the output will be the same as the input.
#       See: https://intellipaat.com/community/558/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-maxpool-of-tensorflow
# - Filter size:
#       https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
#       Notice at as our output spatial volume is decreasing our number of filters learned is increasing — this is a common practice in designing CNN architectures and one I recommend you do as well. As far as choosing the appropriate number of filters
#       , I nearly always recommend using powers of 2 as the values.
#       You may need to tune the exact value depending on (1) the complexity of your dataset and (2) the depth of your neural network, but I recommend starting with filters in the range [32, 64, 128] in the earlier and increasing up to [256, 512, 1024] in the deeper layers.



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


#Using a LR Annealer
# - reduces learning rate when loss stops improving:
# - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
#Note: Need to add this to the model.fit as a call back

batch_size=128
epochs=50

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)


#data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


#compiling and summary

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Fitting on the Training set and making predictions on the Validation set
#using fit_generator() - https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

# Evaluating the Model Performance

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#Visualizing Predictons on the Validation Set

# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
    
    
#CORRECTLY CLASSIFIED FLOWER IMAGES

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)

for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        #ax[i,j].set_title("Predicted Flower : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform(np.argmax([y_test[prop_class[count]]]))))
        plt.tight_layout()
        count+=1
        
        
#MISCLASSIFIED IMAGES OF FLOWERS

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        #ax[i,j].set_title("Predicted Flower : "+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform(np.argmax([y_test[mis_class[count]]]))))
        plt.tight_layout()
        count+=1
        
        





