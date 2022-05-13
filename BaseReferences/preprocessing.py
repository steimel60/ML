#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#--------- Load Data ------------#
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer() #Load data into variable
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1) #split into testing and training data

#Obeserve shape of data
print(X_train.shape)
print(X_test.shape)

#-------- Preprocessing ----------#
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() #instance of preporcessing class
scaler.fit(X_train)     #fit to training data - calculates transform

X_train_scaled = scaler.transform(X_train)   #Actually scale the data

#print(f"Feature Mins before scaling:\n{X_train.min(axis=0)}")
#print(f"Feature Maxes before scaling:\n{X_train.max(axis=0)}")

#print(f"Feature Mins after scaling:\n{X_train_scaled.min(axis=0)}")
#print(f"Feature Maxes after scaling:\n{X_train_scaled.max(axis=0)}")

#We also need to transform the test set
X_test_scaled = scaler.transform(X_test)    #Apply transform to test data

#-------- Effects of Scaling ----------#
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0) #split into testing and training data

svm = SVC(C=100).fit(X_train, y_train)
print(f"Not Preprocessed Test Score: {svm.score(X_test, y_test)}")

scaler = MinMaxScaler() #instance of preporcessing class
scaler.fit(X_train)     #fit to training data - calculates transform

X_train_scaled = scaler.transform(X_train)   #Actually scale the data
X_test_scaled = scaler.transform(X_test)     #Apply transform to test data

svm.fit(X_train_scaled, y_train)             #Fit to scaled data
print(f"Processed Test Score: {svm.score(X_test_scaled, y_test)}")
