
# coding: utf-8

# In[1]:

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
import keras
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input


# In[2]:

read_data = pd.read_csv(sys.arg[1],encoding='big5')
read_data = np.asarray(read_data)
x_read_data = read_data[:,1]
x_read_data.shape

x_data = []
for i in range(x_read_data.shape[0]):
    temp = x_read_data[i].split()
    for j in temp:
        x_data.append(j)
x_data = np.asarray(x_data)
x_data = x_data.reshape(28709,48,48,1).astype(int)

y_data = read_data[:,0]
y_data = y_data.astype(int)


# In[ ]:

batch_size = 64
num_classes = 7 
epochs = 80 
data_augmentation = True 
x_train = x_data 
y_train = y_data 
# The data, shuffled and split between train and test sets: 
#print('x_train shape:', x_train.shape) 
#print(x_train.shape[0], 'train samples') 
#print(x_test.shape[0], 'test samples') 

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes) 
#y_test = keras.utils.to_categorical(y_test, num_classes) 
model = Sequential() 
model.add(Conv2D(64, (3, 3), activation='relu',padding='same', input_shape=x_train.shape[1:])) 
model.add(Activation('relu')) 
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu',padding='same', input_shape=x_train.shape[1:])) 
model.add(Activation('relu')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3))) 
model.add(Activation('relu')) 
model.add(BatchNormalization()) 
model.add(Conv2D(128, (3, 3))) 
model.add(Activation('relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2)) 
model.add(Conv2D(256, (3, 3), activation='relu',padding='same', input_shape=x_train.shape[1:])) 
model.add(Activation('relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(256, (3, 3),activation='relu', padding='same', input_shape=x_train.shape[1:])) 
model.add(Activation('relu')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Conv2D(512, (3, 3),activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(512, (3, 3))) 
model.add(Activation('relu')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(1024)) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.45)) 
model.add(Dense(1024)) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(0.45)) 
model.add(Dense(7)) 
model.add(Activation('softmax'))
# initiate RMSprop optimizer 
opt = keras.optimizers.adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
# Let's train the model using RMSprop 
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
x_train = x_train.astype('float32') 
#x_test = x_test.astype('float32')
x_train /= 255 
#x_test /= 255 


if not data_augmentation:
    #print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=None
              )
else:
    #print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs
                       )


# In[ ]:

model.save("0.65924.h5")
model.save_weights("0.65924_weights.h5")

