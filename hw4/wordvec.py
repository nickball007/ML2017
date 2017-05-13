
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


# In[9]:

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


# TA sample code

# In[ ]:

from argparse import ArgumentParser
import word2vec
import numpy as np
import nltk


# In[ ]:

parser = ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='Set this flag to train word2vec model')
parser.add_argument('--corpus-path', type=str, default='hp/all',
                    help='Text file for training')
parser.add_argument('--model-path', type=str, default='hp/model.bin',
                    help='Path to save word2vec model')
parser.add_argument('--plot-num', type=int, default=500,
                    help='Number of words to perform dimensionality reduction')
args = parser.parse_args()


if args.train:
    # DEFINE your parameters for training
    MIN_COUNT = 0
    WORDVEC_DIM = 0
    WINDOW = 0
    NEGATIVE_SAMPLES = 0
    ITERATIONS = 0
    MODEL = 1
    LEARNING_RATE = np.nan

    # train model
    word2vec.word2vec(
        train=args.corpus_path,
        output=args.model_path,
        cbow=MODEL,
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
        verbose=True)
else:
    # load model for plotting
    model = word2vec.load(args.model_path)

    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:args.plot_num]
    vocabs = vocabs[:args.plot_num]

    '''
    Dimensionality Reduction
    '''
    # from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)


    '''
    Plotting
    '''
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    # plt.savefig('hp.png', dpi=600)
    plt.show()


# In[4]:




# In[ ]:



