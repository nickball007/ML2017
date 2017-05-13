
# coding: utf-8

# In[1]:

import word2vec
import nltk
from sklearn.manifold import TSNE
import numpy as np
#from nltk import av
#from nltk import maxent_treebank_pos_tagger
from nltk import punkt
import sklearn.manifold.t_sne
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
#nltk.download()


# In[61]:

word2vec.word2vec(
    train='all.txt',
    output='/Users/nickball007/ML2017/hw4/wordmodel5.bin',
    size=130,
    window=5,
    sample='1e-5',
    hs=1,
    negative=6,
    threads=os.cpu_count(),
    iter_=7,
    min_count=5,
    alpha=0.025,
    debug=0,
    binary=1,
    cbow=1,
    save_vocab=None,
    read_vocab=None,
    verbose=False)


# In[62]:

model = word2vec.load("wordmodel4.bin")


# In[63]:

vocabs = []
word_vec = []

for vocab in model.vocab:
    vocabs.append(vocab)
    word_vec.append(model[vocab])

vocabs = vocabs[:500]
word_vec = np.array(vecs)[:500]


# In[64]:

tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0,
            n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean',
            init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
    
reduced = tsne.fit_transform(word_vec)


# In[65]:

use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ['<', '/', '>', '—', '|', '-', '.', '“', '”', ',', '’', '?', '!', ';', '"', ':', '‘', '(', '\\', '•', ')', '■']

plt.figure()
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.savefig('hp4.png')

