#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#------------------- generate data ----------------------#

#Classification Data
X, y = mglearn.datasets.make_forge() #2 class classification dataset w/ 2 features
#mglearn.discrete_scatter(X[:,0],X[:,1],y) #plot data set
#plt.legend(['Class 0','Class 1'], loc=4)
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.show()

#Regression Data
X, y = mglearn.datasets.make_wave(n_samples=40) #Regression dataset
#plt.plot(X,y,'o')
#plt.ylim(-3,3)
#plt.xlabel('Feature')
#plt.ylabel('Target')
#plt.show()

#Real-World Classification Example
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
#print({n : v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}) #See classification breakdown
#print(cancer.feature_names) #see list of feature names

#Real-World Regression Example
from sklearn.datasets import load_boston

boston = load_boston() #Get data set on housing prices in boston
X, y = mglearn.datasets.load_extended_boston() #Get "feature engineered" boston dataset
