
# coding: utf-8

# In[1]:

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
import keras
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.applications.imagenet_utils import preprocess_input
import sys


# In[2]:

model_path = '0.65924.h5'
model = load_model(model_path)


# In[3]:

x_test_read_data = pd.read_csv(sys.arg[1],encoding='big5')
x_test_read_data = np.asarray(x_test_read_data)
x_test_read_data = x_test_read_data[:,1]

x_test_data = []
for i in range(x_test_read_data.shape[0]):
    temp = x_test_read_data[i].split()
    for j in temp:
        x_test_data.append(j)
x_test_data = np.asarray(x_test_data)
x_test_data = x_test_data.reshape(7178,48,48,1).astype(int)
x_test_data = x_test_data.astype('float32')
x_test_data /= 255


# In[4]:

y_test = model.predict(x_test_data,batch_size = 64, verbose = 1)
y_test = np.argmax(y_test, axis = 1)

result = []
for i in range(y_test.shape[0]):
    result.append(i)
    result.append(y_test[i])
result = np.asarray(result).reshape(y_test.shape[0],2)

out = pd.DataFrame(data = result,columns = ['id','label'])
out[['id']] = out[['id']].astype(int)

out.to_csv(sys.arg[2], index=False)


# In[ ]:



