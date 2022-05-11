#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import os

#Load Movie Reviews
from sklearn.datasets import load_files

#Download url: https://ai.stanford.edu/~amaas/data/sentiment/
reviews_train = load_files("D:/Dylan/Documents/MovieReviewDatabase/train")

text_train, y_train = reviews_train.data, reviews_train.target
print(f"type(text_train): {type(text_train)}")
print(f"Length: {len(text_train)}")
print(f"text_train[6]:\n{text_train[6]}")

#replace html formatting
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

#load test_data
reviews_test = load_files("D:/Dylan/Documents/MovieReviewDatabase/test")
text_test, y_test = reviews_test.data, reviews_test.target
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]



#------------------ Bag of Words ------------------------#
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Quick Example
bards_words = ["The fool doth think he is wise,","but the wise man knows himself to be a fool"]

vect = CountVectorizer()
vect.fit(bards_words)
print(f"Vocabulary size: {len(vect.vocabulary_)}")
print(f"Vocabulary content:\n {vect.vocabulary_}")
bag_of_words = vect.transform(bards_words)
print(f"bag_of_words: {repr(bag_of_words)}")
print(f"bag_of_words as array: {bag_of_words.toarray()}")

#Movie Example using data from above
vect = CountVectorizer().fit(text_train) #Build vocabulary from all the reviews
X_train = vect.transform(text_train)     #Transform into a sparse array
X_test = vect.transform(text_test)       #also transform test data
print(f"X_train: {repr(X_train)}")
feature_names = vect.get_feature_names() #Extract features
print(f"Number of features: {len(feature_names)}")
print(f"First 20 features: {feature_names[:20]}")
#Build model to predict if a review is positive or negative based on bag of words
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print(f"Mean cross val score: {np.mean(scores)}")
#Play with params
param_grid = {'C':[.001, .01, .1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best Cross Val Score: {grid.best_score_}")
print(f"Best Params: {grid.best_params_}")
print(f"Test Score: {gris.score(X_test, y_test)}")

#We can also set the min number of documents that must contain a "word" to have that word added to our vocabulary
#This helps get rid of typos or unhelpful words
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)

#-------------------- Stopwords ---------------------------#
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

"""
Stop words are words that are very commonly found in a language.
Removing them can sometimes increase a models accuracy.
"""

print(list(ENGLISH_STOP_WORDS[::10]))

#------------------- TF IDF -----------------------------#
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

"""
This gives weights to words based on uniquness to a document.
If a word appears many times in one document but rarely in any other it most likely describes that documents content well.
"""

pipe = make_pipeline(TfidfVectorizer(min_df=5). LogisticRegression())
param_grid = {'logisticregression__C':[.001, .01, .1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print(f"Tfid Score: {grid.best_score_}")
