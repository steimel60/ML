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
from sklearn.linear_model import LogisticRegression

#Build pipeline
pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
pipe.fit(cancer.data)
#Get the 2 principal components from step 2
components = pipe.named_steps["pca"].components_
print(components)


#We can also do this with GridSearchCV
pipe = make_pipeline(StandardScaler(), LogisticRegression()) #Make pipeline
param_grid = {'logisticregression__C':[.01, .1, 1, 10, 100]} #Make parameter grid
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4) #Split data
grid = GridSearchCV(pipe, param_grid, cv=5) #Grid Search over pipeline with 5 folds
grid.fit(X_train, y_train) #Fit our GridSearchCV
print(grid.best_estimator_)#Our best estimator is a pipeline with 2 steps
print(grid.best_estimator_.named_steps["logisticregression"]) #So we can get steps from it
print(grid.best_estimator_.named_steps["logisticregression"].coef_) #And other attributes


#--------------- Preprocessing Pipelines ------------------#
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

pipe = make_pipeline(StandardScaler(),PolynomialFeatures(),Ridge())
param_grid = {'polynomialfeatures__degree':[1, 2, 3], 'ridge__alpha':[.001,.01,.1,1,10,100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
#Show results on heatmap
plt.matshow(grid.cv_results_['mean_test_score'].reshape(3,-1), vmin=0, cmap="viridis")
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree'])
plt.colorbar()
plt.show()
