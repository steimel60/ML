#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#------------------- generate data ----------------------#
from sklearn.model_selection import train_test_split

#Classification Data
X, y = mglearn.datasets.make_forge() #2 class classification dataset w/ 2 features
#Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#---------------------- Build Model ---------------------#
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3) #initiate model
clf.fit(X_train,y_train) #fit to training data

#----------------- Test/Evaluate Predictions -----------------#

predictions = clf.predict(X_test) #make predictions on our testing data
acc = clf.score(X_test, y_test) #get model accuracy on test data
#print(acc)

#-------------- Visualize Class Boundaries --------------#

#fig, axes = plt.subplots(1,3, figsize=(10,3))

#for k, ax in zip([1,3,9], axes):
#    clf = KNeighborsClassifier(n_neighbors=k).fit(X,y) #init and fit in one line :)
    #plot
#    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=.5, ax=ax, alpha=.4)
#    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
#    ax.set_title(f"{k} Neighbor(s)")
#    ax.set_xlabel("feature 0")
#    ax.set_ylabel("feature 1")
#plt.show()

#----------------- Real World Example ---------------------#
from sklearn.datasets import load_breast_cancer

#Load Data
cancer = load_breast_cancer()
#Split data
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)

train_acc = []
test_acc = []

for k in range(1,11):
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    train_acc.append(clf.score(X_train,y_train))
    test_acc.append(clf.score(X_test,y_test))

#plt.plot([k for k in range(1,11)], train_acc, label="Training Accuracy")
#plt.plot([k for k in range(1,11)], test_acc, label="Testing Accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("K Neighbors")
#plt.legend()
#plt.show()

#--------------- Knn Regression Example ------------------#
from sklearn.neighbors import KNeighborsRegressor

#Quick Visual
#mglearn.plots.plot_knn_regression(n_neighbors=3)
#plt.show()

X,y = mglearn.datasets.make_wave(n_samples=40) #generate data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0) #split data

reg = KNeighborsRegressor(n_neighbors=3).fit(X_train,y_train)   #initiate and fit model

predictions = reg.predict(X_test)
acc = reg.score(X_test,y_test)
#print(acc)

#------------------ Analyizing Regressor -----------------#

fig, axes = plt.subplots(1,3, figsize=(15,4))

#create 1000 data points evenly spaced from -3 to 3
line = np.linspace(-3,3,1000).reshape(-1,1)

for k, ax in zip([1,3,9], axes):
    reg = KNeighborsRegressor(n_neighbors=k).fit(X_train,y_train)
    ax.plot(line, reg.predict(line)) #Plot predictions for points in "line"
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0),markersize=8) #Plot the training data
    ax.plot(X_test, y_test, 'o', c=mglearn.cm2(1),markersize=8) #Plot the testing data

    ax.set_title(f"{k} Neighbors. Train Score: {round(reg.score(X_train,y_train),2)}  Test Score: {round(reg.score(X_test,y_test),2)}")
axes[0].legend(["Model Predictions","Training data/target","Test data/target"], loc='best')
plt.show()
