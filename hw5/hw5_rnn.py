
# coding: utf-8

# In[ ]:

import numpy as np

import sys
import keras.backend as K 
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Dense,Dropout, GRU
#from keras.optimizers import Adam
#from keras import optimizers
from keras.models import load_model
#import pandas as pd
#from keras.models import Sequential, Model
#from keras.preprocessing import sequence
#import keras
#from keras.preprocessing import text
#from sklearn.preprocessing import MultiLabelBinarizer 
#from sklearn.metrics.classification import matthews_corrcoef
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

train_path = "train_data.csv"
test_path = "test_data.csv"
output_path = "output2.csv"
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 200

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding="utf-8") as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

tag_list = []
with open('051965token.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('taglist.pkl', 'rb') as f:
    tag_list = pickle.load(f)
model = load_model('051965best.hdf5', custom_objects={'f1_score': f1_score})
(_, X_test,_) = read_data(sys.argv[1],False)

word_index = tokenizer.word_index

test_sequences = tokenizer.texts_to_sequences(X_test)

test_sequences = pad_sequences(test_sequences,maxlen=306)



Y_pred = model.predict(test_sequences)
thresh = 0.4
with open(sys.argv[2],'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

