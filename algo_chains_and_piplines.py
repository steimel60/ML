#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import os

#------------------- SVM on Cancer Data --------------------------#
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Load the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
#Scale the data
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Build the model
svm = SVC()
svm.fit(X_train_scaled, y_train)
#Check score
print(f"Test Score: {svm.score(X_test_scaled, y_test)}")

"""
If we were to use "X_train_scaled" to fit grid search we would have validaiton predictions on scaled data.
Followed by test predictions on non-scaled data - this is bad.
We can use the Pipeline class to "glue" together multiple processes to avoid this issue.
"""

#------------------- Building Pipelines ---------------------------#
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

#Creates 2 steps, scaling with MinMaxScaler, and an SVC model
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

#We can fit pipelines like any other sklearn estimator
pipe.fit(X_train, y_train) #Fit non-scaled data from cancer dataset

#Use score method to get the score of our pipeline
print(f"Pipeline Test Score: {pipe.score(X_test, y_test)}")

"""
We can use the pipline to use a single estimator in our GridSearchCV.
We create a param_grid dictionary like normal, with a specific syntax to tell it which step of the pipeline to use.
"""
#This dict tells GridSearchCV to use these parameter lists in our "svm" pipeline step
param_grid = {'svm__C': [.001, .01, .1, 1, 10, 100],
                'svm__gamma': [.001, .01, .1, 1, 10, 100]}
#Use GridSearchCV as usual
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best CV accuracy: {grid.best_score_}")
print(f"Best parameters: {grid.best_params_}")
print(f"Test Set Score: {grid.score(X_test, y_test)}")

#make_pipeline can simplify this process further by removing the requirement of user named steps
easy_pipe = make_pipeline(MinMaxScaler(),SVC(C=100))
print(f"make_pipeline Steps:\n{easy_pipe.steps}")


#-------------------- Accessing Attributes ----------------------------#
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
pipe.fit(cancer.data)
#Get the 2 principal components from step 2
components = pipe.named_steps["pca"].components_
print(components)
