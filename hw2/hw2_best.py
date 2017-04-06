
# coding: utf-8

# In[ ]:

import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
train_x = pd.read_csv(sys.argv[3],encoding='big5')
train_y = pd.read_csv(sys.argv[4], encoding = 'big5', header = None)
w = np.zeros((107,1))
a = np.ones((32561,1))
x = np.matrix(train_x, dtype = 'float64')
x = np.asarray(x)

mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
index = [0, 1, 3, 4, 5]
mean_vec = np.zeros(x.shape[1])
std_vec = np.ones(x.shape[1])

mean_vec[index] = mean[index]
std_vec[index] = std[index]
X_normed = (x - mean_vec) / std_vec

y = np.matrix(train_y, dtype = 'float64')
x = np.hstack((a, X_normed))
pre_gra = 0
lr = 1
iteration = 3000
lamda = 0

for p in range(iteration):
    gradient = np.zeros_like(w)
    z = x.dot(w)
    z=np.matrix(z)
    sigmoid = 1/ (1 + np.exp(-z))
    gradient = (((sigmoid - y).T).dot(x)).T + 2* lamda * w
    pre_gra += np.square(gradient)
    adg = np.sqrt(pre_gra)
    w = np.subtract(w, np.multiply((np.divide(lr,adg)), gradient))
    res = 1/ (1 + np.exp(-(x.dot(w))))

percentage = 0
for i in range(len(res)):
    if res[i] >= 0.5:
        if y[i,0] == 1:
            percentage += 1
    else:
        if y[i,0] == 0:
            percentage += 1
res_percentage = percentage / y.shape[0]
test_x = pd.read_csv(sys.argv[5],encoding='big5')
test_x = test_x.replace('nan', 0)
test_x = np.matrix(test_x, dtype='float64')
test_x = np.asarray(test_x)


#scaling
test_x_mean = np.mean(test_x, axis=0)
test_x_std = np.std(test_x, axis=0)
index = [0, 1, 3, 4, 5]
mean_vec1 = np.zeros(test_x.shape[1])
std_vec1 = np.ones(test_x.shape[1])

mean_vec1[index] = test_x_mean[index]
std_vec1[index] = test_x_std[index]

X_test_normed = (test_x - mean_vec1) / std_vec1
b = np.ones((16281,1))
test_x = np.hstack((b, X_test_normed))

# np.isnan(test_x).any()

zz = test_x.dot(w)

sigmoid1 = 1/ (1+ np.exp(-zz))

test_y = []

for i in sigmoid1:
    if i >= 0.5:
        test_y.append(1)
    else:
        test_y.append(0)

test_y = np.matrix(test_y).T

num_rows, num_cols = test_y.shape
id_value = []
for i in range(num_rows):
    id_value.append(i+1)
id_value = np.matrix(id_value).T
predict = np.hstack((id_value, test_y))
df = pd.DataFrame(data = predict,columns = ['id','label'])
df.to_csv(sys.argv[6], index = False)

