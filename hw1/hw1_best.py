
# coding: utf-8

# In[1]:

import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

data = pd.read_csv(sys.argv[1],encoding='big5')
data = data.replace('NR', 0)

train_Rdata = data.drop(data.columns[[0, 1, 2]], axis=1).apply(pd.to_numeric).as_matrix()


# In[2]:

train_Rdata = data.drop(data.columns[[0, 1, 2]], axis=1).apply(pd.to_numeric).as_matrix()
data1 = train_Rdata[0::18,:].reshape(1, 5760)
data2 = train_Rdata[1::18,:].reshape(1, 5760)
data3 = train_Rdata[2::18,:].reshape(1, 5760)
data4 = train_Rdata[3::18,:].reshape(1, 5760)
data5 = train_Rdata[4::18,:].reshape(1, 5760)
data6 = train_Rdata[5::18,:].reshape(1, 5760)
data7 = train_Rdata[6::18,:].reshape(1, 5760)
data8 = train_Rdata[7::18,:].reshape(1, 5760)
data9 = train_Rdata[8::18,:].reshape(1, 5760)
data10 = train_Rdata[9::18,:].reshape(1, 5760)
data11 = train_Rdata[10::18,:].reshape(1, 5760)
data12 = train_Rdata[11::18,:].reshape(1, 5760)
data13 = train_Rdata[12::18,:].reshape(1, 5760)
data14 = train_Rdata[13::18,:].reshape(1, 5760)
data15 = train_Rdata[14::18,:].reshape(1, 5760)
data16 = train_Rdata[15::18,:].reshape(1, 5760)
data17 = train_Rdata[16::18,:].reshape(1, 5760)
data18 = train_Rdata[17::18,:].reshape(1, 5760)

train_data = np.vstack((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18))


# In[3]:

train_x = [] #features 共5751組 一組有163個 第一個是1跟bias相乘
train_y = [] #PM2.5 answer

train_y.append(train_data[9, 9:])
train_y= np.asarray(train_y).reshape(5751, 1)

for t in range(5751):
    train_x.append(1)
    for j in range(18):
        for i in range(9):
            train_x.append(train_data[j,i+t])

train_x = np.asarray(train_x).reshape(5751, 163)


# In[4]:

wei = np.zeros((163,1))
lr = 0.05
iteration = 80000
pre_gra = 0

for p in range(iteration):
    gradient = np.zeros_like(wei)
    gradient = (2*(train_x.dot(wei) - train_y).T.dot(train_x).T)
    pre_gra += gradient**2
    adg = np.sqrt(pre_gra)
    wei = wei - lr/adg*gradient


# In[5]:

test_data = pd.read_csv(sys.argv[2], encoding='big5', header = None)
test_data = test_data.replace('NR', 0)
test_data1 = np.matrix((test_data.values[:, 2:]), dtype='float64')
a = np.ones((240,1))
test_data1 = np.hstack((a, test_data1.reshape(240,162)))
predict = test_data1.dot(wei)
idvalue = test_data.ix[1::18, 0]
idvalue = idvalue.reshape((240,1))
predict = np.hstack((idvalue,predict))
df = pd.DataFrame(data = predict,columns = ['id','value'])
df
df.to_csv(sys.argv[3], index = False)


# In[ ]:



