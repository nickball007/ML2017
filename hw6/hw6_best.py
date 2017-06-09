from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Flatten, Concatenate, Input, BatchNormalization, Dot, Add
from keras.models import Sequential, Model
import os
import sys

MODEL_WEIGHTS_FILE = '085117.h5'
K_FACTORS = 400
RNG_SEED = 1

test_data = pd.read_csv(os.path.join(sys.argv[1],'test.csv'))
test = np.asarray(test_data)

class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Dropout(0.2))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Dropout(0.2))
        Q.add(Reshape((k_factors,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))
#dot => concat, 
#model.add(Dense(128))
#
#model.add(Dense(1))
#

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

trained_model = CFModel(6040, 3952, 400)
trained_model.load_weights(MODEL_WEIGHTS_FILE)
answer = trained_model.predict([test[:,1],test[:,2]])
with open(sys.argv[2],'w') as output:
    print ('\"TestDataID\",\"Rating\"',file=output)
    for i in range(len(answer)):
        Rating = answer[i][0]
        print ('\"%d\",\"%s\"'%(i+1,Rating),file=output)