import re,sys
from nltk.corpus import stopwords
import nltk
import string
import numpy as np
from nltk.stem.porter import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

import scipy.sparse

import sklearn.linear_model
import sklearn.datasets
import sklearn.svm
import sklearn.metrics
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.utils.sparsefuncs
DATASET_DIR = "/home/mitesh/Documents/MS Hackathon/BingHackathonTrainingData.txt"
OUTPUT_DIR = "/home/mitesh/Documents/MS Hackathon/Output_Reg.txt"
TEST_FILE = "/home/mitesh/Documents/MS Hackathon/BingHackathonTestData.txt"


author_dict = {}
ll = []
year_vec = []
with open(DATASET_DIR, 'r') as f:
    for line in f:
        tab_sep = line.split('\t')
        # 0 - id, 1 - topic_id(class), 2 - year, 3 - authors, 4 - title,5- sentences
        doc_id = tab_sep[0]
        topic_id = tab_sep[1]
        year = tab_sep[2]
        authors = tab_sep[3]
        title_line = tab_sep[4]
        sentences = tab_sep[5]
        line_ = title_line.split(' ')
        list_ = []
        year_vec.append(float(year))
        for word in line_:
            list_.append(str(word))
        line_ = str(sentences).split(' ')
        for word in line_:
            list_.append(str(word))
        for author in authors.split(';'):
            a = author.split(' ')
            s = ""
            for b in a:
                s += b
            list_.append(str(s))
        ll.append(' '.join(list_))
            
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             dtype='double',
                             max_features = 5000) 


train_data_features = vectorizer.fit_transform(ll)
vectors =  train_data_features
pca = sklearn.decomposition.TruncatedSVD(n_components=90)
data = pca.fit_transform(train_data_features)

regression = sklearn.linear_model.LinearRegression()
regression.fit(data, year_vec)

ll = []
year_vec = []
with open(TEST_FILE, 'r') as f:
    for line in f:
        tab_sep = line.split('\t')
        # 0 - id, 1 - topic_id(class), 2 - year, 3 - authors, 4 - title,5- sentences
        doc_id = tab_sep[0]
        topic_id = tab_sep[1]
        year = tab_sep[2]
        authors = tab_sep[3]
        title_line = tab_sep[4]
        sentences = tab_sep[5]
        line_ = title_line.split(' ')
        list_ = []
        year_vec.append(float(year))
        for word in line_:
            list_.append(str(word))
        line_ = str(sentences).split(' ')
        for word in line_:
            list_.append(str(word))
        for author in authors.split(';'):
            a = author.split(' ')
            s = ""
            for b in a:
                s += b
            list_.append(str(s))
        ll.append(' '.join(list_))
            
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(ll)
pca = sklearn.decomposition.TruncatedSVD(n_components=90)
data = pca.fit_transform(train_data_features)
y = regression.predict(data )

with open(OUTPUT_DIR, 'a') as f:
    for i, pred in enumerate(y):
        f.write("{}\t{}\n".format(i+5001, pred))
    
    


"""
y = forest.predict(train_data_features)
for i, pred in enumerate(y):
    print i+5001, pred
"""
