
# coding: utf-8

# In[ ]:

# datasetX, principle components U, the numbers of dimensions to reduce to k


# In[1]:

import numpy as np
# sklearn are prohibited
import os
#from sklearn.decomposition import PCA
#import sklearn
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# # read CMU AMP lab facial expression database

# In[2]:

import os
from scipy import misc
path = '/Users/nickball007/ML2017/hw4/faceExpressionDatabase/'
#image= misc.imread(path + filenamelist[0], flatten= 0)

filenamelist = []
for c in range(ord('A'), ord('J')+1):
    for j in range(10):
        filename = chr(c) + "0" + str(j) + ".bmp"
        filenamelist.append(filename)
        
imagelist = []
for i in range(len(filenamelist)):
    image = misc.imread(path + filenamelist[i], flatten= 0)
    imagelist.append(image)
# imagelist 100張圖的(64,64)


# 1. 

# In[3]:

# image平均 (64,64)
image_avg = np.zeros((64,64))
temp = np.zeros(100)
#image_avg = np.asmatrix(image_avg)
for i in range(64):
    for j in range(64):
        for k in range(100):
            temp[k] = imagelist[k][i][j]
            image_avg[i][j] = np.average(temp) 


# In[4]:

#image的行向量
imagelist_Column_vector = np.zeros((100,4096))
for i in range(100):
    for j in range(64):
        for k in range(64):
            imagelist_Column_vector[i][64*j+k] = imagelist[i][j][k]
    
#imagelist_Column_vector.shape => (100, 4096) 
#imagelist_Column_vector為100張圖片的行向量


# In[5]:

#image平均的行向量 image_avg_Column_vector = (4096,)
image_avg_Column_vector = np.zeros((4096))
for i in range(64):
    for j in range(64):
        image_avg_Column_vector[64*i + j] = image_avg[i][j]


# 1.1 top 9 eigenfaces

# In[10]:

#減過平均的100個4096行向量imagelist_Column_vector_minus_avg算出Cov matrix計算eigenvalue, eigenvector
imagelist_Column_vector_minus_avg = imagelist_Column_vector
for i in range(100):
    imagelist_Column_vector_minus_avg[i] = imagelist_Column_vector[i] - image_avg_Column_vector
S = imagelist_Column_vector_minus_avg.transpose().dot(imagelist_Column_vector_minus_avg)
eig_val_cov1, eig_vec_cov1= np.linalg.eigh(S)


# In[11]:

#SVD
u, s, v = np.linalg.svd(imagelist_Column_vector_minus_avg)


# In[12]:

plott = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        plott[i][j] =v[1][i*64+j]
        
plt.clf()
plt.imshow(image_avg)
plt.gray()
plt.show()


# In[13]:

sec1 = np.zeros((64,64))#4095
sec2 = np.zeros((64,64))#4094
sec3 = np.zeros((64,64))#4093
sec4 = np.zeros((64,64))#4092
sec5 = np.zeros((64,64))#4091
sec6 = np.zeros((64,64))#4090
sec7 = np.zeros((64,64))#4089
sec8 = np.zeros((64,64))#4088
sec9 = np.zeros((64,64))#4087
eigenface = np.zeros((192,192))


# In[14]:

# top 9 eigenfaces
for i in range(64):
    for j in range(64):
        sec1[i][j] =v[0][i*64+j]
for i in range(64):
    for j in range(64):
        sec2[i][j] =v[1][i*64+j]
for i in range(64):
    for j in range(64):
        sec3[i][j] =v[2][i*64+j]
for i in range(64):
    for j in range(64):
        sec4[i][j] =v[3][i*64+j]
for i in range(64):
    for j in range(64):
        sec5[i][j] =v[4][i*64+j]
for i in range(64):
    for j in range(64):
        sec6[i][j] =v[5][i*64+j]
for i in range(64):
    for j in range(64):
        sec7[i][j] =v[6][i*64+j]
for i in range(64):
    for j in range(64):
        sec8[i][j] =v[7][i*64+j]
for i in range(64):
    for j in range(64):
        sec9[i][j] =v[8][i*64+j]      


# In[15]:

sec9.shape


# In[16]:

plt.subplot(331)
plt.plot(sec1)
plt.imshow(sec1)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(332)
plt.plot(sec2)
plt.imshow(sec2)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(333)
plt.imshow(sec3)
plt.plot(sec3)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(334)
plt.plot(sec4)
plt.imshow(sec4)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(335)
plt.plot(sec5)
plt.imshow(sec5)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(336)
plt.plot(sec6)
plt.imshow(sec6)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(337)
plt.plot(sec7)
plt.imshow(sec7)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(338)
plt.plot(sec8)
plt.imshow(sec8)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()
plt.subplot(339)
plt.imshow(sec9)
plt.plot(sec9)
plt.axis([63,0,63,0])
plt.axis('off')
plt.gray()

plt.show()


# 1.2 

# In[17]:

#top 5 eigenface
thirdanswer = []
weightlist = []
for i in range(0,100):
    temp1 = np.dot(imagelist_Column_vector_minus_avg[i],v[0])*v[0]
    temp2 = np.dot(imagelist_Column_vector_minus_avg[i],v[1])*v[1]
    temp3 = np.dot(imagelist_Column_vector_minus_avg[i],v[2])*v[2]
    temp4 = np.dot(imagelist_Column_vector_minus_avg[i],v[3])*v[3]
    temp5 = np.dot(imagelist_Column_vector_minus_avg[i],v[4])*v[4]
    temp = temp1 + temp2 + temp3 + temp4 + temp5
    weight = np.zeros((64,64))
    for j in range(64):
        for k in range(64):
            weight[j][k] =temp[j*64+k]
    thirdanswer.append(weight)
    answerweight = weight + image_avg
    weightlist.append(answerweight)   


# In[18]:

#reconstruct eigenface
for i in range(1,11):
    for j in range(1,11):
        #secondans.add_subplot(i,j,i*10+j)
        plt.subplot(10,10,(i-1)*10+j)
        plt.imshow(weightlist[(i-1)*10+(j-1)])
        #plt.plot(weightlist[(i-1)*9+(j-1)])
        plt.axis([63,0,63,0])
        plt.axis('off')
        plt.gray()
plt.show()


# In[19]:

#dataset 前100
for i in range(1,11):
    for j in range(1,11):
        #secondans.add_subplot(i,j,i*10+j)
        plt.subplot(10,10,(i-1)*10+j)
        plt.imshow(imagelist[(i-1)*10+(j-1)])
        #plt.plot(weightlist[(i-1)*9+(j-1)])
        plt.axis([63,0,63,0])
        plt.axis('off')
        plt.gray()
plt.show()


# 1.3

# In[20]:

rmse = np.zeros((64,64))
rmselist = []
for k in range(100):
    rmse = imagelist[k] - weightlist[k]
    rmselist.append(rmse)


# In[22]:

#top x eigenface <=0.01
thirdanswer = []
weightlist = []
  
for i in range(0,100):
    temp = np.dot(imagelist_Column_vector_minus_avg[i],v[0])*v[0]
    for l in range(1,60):
        temp = temp + np.dot(imagelist_Column_vector_minus_avg[i],v[l])*v[l]
        
    weight = np.zeros((64,64))
    for j in range(64):
        for k in range(64):
            weight[j][k] =temp[j*64+k]
    thirdanswer.append(weight)
    answerweight = weight + image_avg
    weightlist.append(answerweight)  
    
rmse = np.zeros((64,64))
rmselist = []
for k in range(100):
    rmse = imagelist[k] - weightlist[k]
    rmselist.append(rmse)
    
temp = np.square(rmselist)
rmsevalue = np.sum(temp)
rmsevalue = rmsevalue/(4096*100)
rmsevalue = np.sqrt(rmsevalue)
rmsevalue = rmsevalue/255
rmsevalue


# In[ ]:



