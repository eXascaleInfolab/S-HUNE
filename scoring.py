# Using DeepWalk evaluation code
#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

__author__      = "Bryan Perozzi"

import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")


import numpy
from scipy.io import loadmat
from scipy.io import savemat
from collections import defaultdict
#import gensim
from itertools import izip
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}

# 0. Files
embeddings_file = str(sys.argv[2])
matfile = str(sys.argv[1])
outputfile = str(sys.argv[3])

# 1. Load Embeddings
# model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
mat = loadmat(embeddings_file)
features_matrix = mat['embs']

# 2. Load labels
mat = loadmat(matfile)
A = mat['network']
graph = sparse2graph(A)
labels_matrix = mat['group']
dim = labels_matrix.shape
# print dim
label_set = list(range(0, dim[1]))
# print label_set
mlb = MultiLabelBinarizer(classes=label_set)

# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
# features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

# 2. Shuffle, to create train/test groups
shuffles = []
number_shuffles = 10
for x in range(number_shuffles):
  shuffles.append(skshuffle(features_matrix, labels_matrix))

# 3. to score each train/test group

#training_percents = [ 0.9]
training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

averages = ["macro", "micro"]  
res = numpy.zeros([number_shuffles, len(training_percents), len(averages)])
  

# uncomment for all training percents
#training_percents = numpy.asarray(range(1,10))*.1
for ii, train_percent in enumerate(training_percents):
  for jj, shuf in enumerate(shuffles):

    X, y = shuf

    training_size = int(train_percent * X.shape[0])

    X_train = X[:training_size, :]
    y_train = y[:training_size]

    # y_train = [[] for x in xrange(y_train_.shape[0])]


    # cy =  y_train_.tocoo()
    # for i, j in izip(cy.row, cy.col):
    #     y_train[i].append(j)

    # assert sum(len(l) for l in y_train) == y_train_.nnz

    X_test = X[training_size:, :]
    y_test_ = y[training_size:]
    
    y_test = [[] for x in xrange(y_test_.shape[0])]

    cy =  y_test_.tocoo()
    for i, j in izip(cy.row, cy.col):
        y_test[i].append(j)

    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train)

    # find out how many labels should be predicted
    top_k_list = [len(l) for l in y_test]
    # print type(top_k_list)
    preds = clf.predict(X_test, top_k_list)
    # print mlb.fit_transform(preds)

    
    for kk,average in enumerate(averages):
        # print MultiLabelBinarizer().fit_transform(y_test).shape
        # print MultiLabelBinarizer().fit_transform(preds).shape
        res[jj,ii,kk] = f1_score(y_test_,  mlb.fit_transform(preds), average=average)

savemat(outputfile,mdict={'res':res})




