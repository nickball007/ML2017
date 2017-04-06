
# coding: utf-8

# In[18]:

import sys
import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# # 讀資料

# In[3]:

Rawdata = pd.read_csv(sys.argv[3],encoding='big5' ) # 原始資料 尚未取feature
X_trainFeature = pd.read_csv("X_train")  #助教給的feature

Y_train = pd.read_csv(sys.argv[4],encoding='big5', header = None) #助教給的Ans


# In[4]:

X_trainFeature = X_trainFeature.values
X_trainFeature = X_trainFeature[:,:]  #將欄位去除, 重複執行會少行, 別亂動
X_trainFeatureData = np.matrix(X_trainFeature,dtype='float64')

#32561筆訓練資料, 106 dim feature X_trainFeatureData.shape = (32561, 106)


# In[5]:

Y_trainData = np.matrix(Y_train,dtype='float64')
#32561筆訓練答案, (32561, 1)


# In[6]:

Data = np.hstack((X_trainFeatureData,Y_trainData))
# Data = (32561,107)


# # class Ans = 1 miu

# class1Data

# In[7]:

k = 0
class1Data = np.zeros((7841,106))

for i in range(32561):
    if Data[i,106] == 1:
        for j in range(106):
            class1Data[k,j] = Data[i,j]
        k = k + 1
        
#class1Data.shape(7841, 106)


# In[8]:

# miu1
miu1 = np.zeros(class1Data.shape[1])

for i in range(106):
    miu1[i] = np.mean(class1Data[:,i])


# # class Ans = 0 miu

# class0Data

# In[9]:

k = 0
class0Data = np.zeros((24720,106))

for i in range(32561):
    if Data[i,106] == 0:
        for j in range(106):
            class0Data[k,j] = Data[i,j]
        k = k + 1
        
#class0Data.shape(24720, 106)


# In[10]:

# miu0
miu0 = np.zeros(class0Data.shape[1])

for i in range(106):
    miu0[i] = np.mean(class0Data[:,i])


# # Covariance

# In[11]:

cov1 = np.cov(class1Data.T)
cov0 = np.cov(class0Data.T)
cov = np.matrix(cov1 * (7841/(7841+24720)) + cov0 * (24720/(7841+24720)))
cov.shape


# # w and b

# In[22]:

cov_I = linalg.pinv(cov)

w = (miu0 - miu1).T.dot(cov_I)
b = -0.5 * miu0.T.dot(cov_I).dot(miu0) + 0.5 * miu1.T.dot(cov_I).dot(miu1) + np.log(24720/7841)

b


# # plot

# In[23]:

test_read_data = pd.read_csv(sys.argv[5], encoding='big5')
test_data = np.asarray(test_read_data.apply(pd.to_numeric).as_matrix())


predict = 1 - 1 / ( 1 + np.exp( - test_data.dot(w.T) - b ))

x = range(len(predict))
plt.plot(x,predict)


# # result

# In[24]:

result = []
for i in range(len(predict)):
    result.append(i+1)
    if predict[i,0] >= 0.5:
        result.append(1)
    else:
        result.append(0)
        
result = np.asarray(result).reshape(len(predict),2)   

out = pd.DataFrame(data = result,columns = ['id','label'])
out[['id']] = out[['id']].astype(int)
out.to_csv(sys.argv[6], index = False)




# In[ ]:



