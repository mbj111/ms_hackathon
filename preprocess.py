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

DATASET_DIR = "/home/mitesh/Documents/MS Hackathon/BingHackathonTrainingData.txt"
OUTPUT_DIR = "/home/mitesh/Documents/MS Hackathon/Output.txt"
TEST_FILE = "/home/mitesh/Documents/MS Hackathon/BingHackathonTestData.txt"


author_dict = {}
ll = []
topic_vec = []
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
        topic_vec.append(topic_id)
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
train_data_features = train_data_features.toarray()

forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( train_data_features, topic_vec )
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)
clf = MultinomialNB().fit(X_train_tfidf, topic_vec)

ll = []
topic_vec = []
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
        topic_vec.append(topic_id)
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
train_data_features = train_data_features.toarray()
X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)
y_clf = clf.predict(X_train_tfidf)

with open(OUTPUT_DIR, 'a') as f:
    for i, pred in enumerate(y_clf):
        f.write("{}\t{}\n".format(i+5001, pred))
    
    


"""
y = forest.predict(train_data_features)
for i, pred in enumerate(y):
    print i+5001, pred
"""
