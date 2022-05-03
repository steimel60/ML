#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import os

#--------------------- Cross Validation ---------------------------#
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

"""
One way of Cross Validating is k-fold cross validation.
This splits the data into k training sets each getting its own model.
We then get the accuracy on the remaining k-1 sets for each model created.
"""
#mglearn.plots.plot_cross_validation() #visualization of k-folds method of splitting data
#plt.show()

iris = load_iris()              #Dataset
logreg = LogisticRegression()   #Model to use
k=5                             #Number of subsets
#Get and show scores of subsets
scores = cross_val_score(logreg, iris.data, iris.target, cv=k)
print(f"Cross Val Scores: {scores}")
#A common way to summarize cross-validation accuracy is with the mean
print(f"Avg CV Score: {scores.mean()}")


#We can use cross_validate() to do something similar but return more info
res = cross_validate(logreg, iris.data, iris.target, cv=k, return_train_score=True)
print("cross_validate() Dict:")
for key in res:
    print(f"{key}: {res[key]}")

"""
It's a good idea to "Stratify" data.
This still creates k subsets but it distributes the subsets over the data, as opposed to taking the first kth part for subset 1 and so on.
This helps prevent situtation where the data may be sorted, say all class 0 is first, and having a subset of only 1 class.
sklearn dies this automatically for classification models
"""
#mglearn.plots.plot_stratified_cross_validation()
#plt.show()

"""
We can also pass more complex infromation into the cv parameter
For example, we can pass a specific split model
"""
kfold = KFold(n_splits=k) #non-stratified splitter
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print(f"Non-Stratified Split k={k} on Classification Data: {scores}")
kfold = KFold(n_splits=3) #non-stratified splitter
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print(f"Non-Stratified Split k=3 on Classification Data: {scores}")
kfold = KFold(n_splits=3, shuffle=True, random_state=0) #shuffled splitter
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print(f"Shuffled Split k=3 on Classification Data: {scores}")
