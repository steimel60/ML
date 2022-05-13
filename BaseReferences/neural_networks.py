#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#---------- Simplified Visual ---------------#
#mglearn.plots.plot_two_hidden_layer_graph()
#plt.show()

#----------- Example Model -------------------#
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100,noise=.25,random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)

mlp = MLPClassifier(solver='lbfgs',random_state=0).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.show()

"""
Reducing the # of hidden layers (default 100) lowers complexity
"""

mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10]).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.show()

#Now with 2 hidden layers
mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.show()

#2 hidden layers with tanh nonlinearity (default is relu)
mlp = MLPClassifier(solver='lbfgs',activation='tanh',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.show()

#Use alpha parameter to shrink weights towards zero
#fig, axes = plt.subplots(2,4, figsize=(20,8))
#for axx, n_hidden_nodes in zip(axes, [10,100]):
#    for ax, alpha in zip(axx,[.001, .01, .1, 1]):
#        mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha).fit(X_train,y_train)
#        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
#        mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
#        ax.set_title(f"n_hidden_nodes = [{n_hidden_nodes},{n_hidden_nodes}]\nalpha={round(alpha,4)}")
#plt.show()


#------------------ Breast Cancer Example ------------------------#
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42).fit(X_train, y_train)
print(f"Train Score: {mlp.score(X_train, y_train)}    Test Score: {mlp.score(X_test, y_test)}")

"""
Neural networks expect input features to have mean of 0 and variance of 1.
We will fix this below and examine results.
"""

mean_on_train = X_train.mean(axis=0) #Compute mean value for each feature
std_on_train = X_train.std(axis=0)   #Compute the standard deviation for each feature

#Subtract mean and scale by inverse std to get Mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
#use the SAME TRANSFORMATION on test set
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0).fit(X_train,y_train)
print(f"Train Score: {mlp.score(X_train, y_train)}    Test Score: {mlp.score(X_test, y_test)}")
