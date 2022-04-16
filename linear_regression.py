#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn


#---------------- Get Data ----------------#
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.make_wave(n_samples=60) #Get data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42) #split into train and test data

#---------------- Build Model --------------#
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train,y_train) #initiate and fit LR model
#print(lr.coef_) #list of coefficeints in LR model, note this only has 1 variable
#print(lr.intercept_) #b value of y = w[0]*x[0]+b
#print(lr.score(X_train,y_train))
#print(lr.score(X_test,y_test))

#------------- Example 2 --------------------#

#This example has a lot more variables, 104
X,y = mglearn.datasets.load_extended_boston() #Get data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0) #split into train and test data

lr = LinearRegression().fit(X_train,y_train) #initiate and fit model
#More features, better results - one feature in example 1 resulted in underfitting
#print(lr.score(X_train,y_train))
#print(lr.score(X_test,y_test))
#print("The large difference in scores is a clear sign of overfitting.")


#--------------- Ridge Regression ------------#
from sklearn.linear_model import Ridge

"""Ridge Regression attempts to make the coefficeints as close to
zero as possible by penalizing the Euclidean length from 0 but otherwise
calculates the same as the above LR model.

This is done with the alpha paramter (default alpha=1). Higher alphas bring
coefficeints closer to zero, lower alphas allow them to be further"""

ridge = Ridge().fit(X_train,y_train)
#print(f'Train Score: {ridge.score(X_train,y_train)}    Test Score: {ridge.score(X_test,y_test)}')

ridge1 = Ridge(alpha=.1).fit(X_train,y_train) #Smaller alpha allows larger coefficeints
#print(f'Train1 Score: {ridge1.score(X_train,y_train)}    Test1 Score: {ridge1.score(X_test,y_test)}')

#----------------- Lasso Model ------------------#
from sklearn.linear_model import Lasso

"""
Similar to ridge a Lasso model restricts coefficeints distance from 0.
The Lasso model however, penalizes the sum of the abs values of the coefficeints.
This results in the coefficeint of some features to be 0, resulting in that feature being ignored.
"""

lasso = Lasso().fit(X_train,y_train)
print(f'Train Score: {lasso.score(X_train,y_train)}    Test Score: {lasso.score(X_test,y_test)}    # of used Features: {np.sum(lasso.coef_ != 0)}')

"""
Poor scores are signs of underfitting - we only used 4 of 104 features.
We can lower alpha and increase max_iter in attempt to fix this.
"""

lasso1 = Lasso(alpha=.01,max_iter=100000).fit(X_train,y_train)
print(f'Train Score: {lasso1.score(X_train,y_train)}    Test Score: {lasso1.score(X_test,y_test)}    # of used Features: {np.sum(lasso1.coef_ != 0)}')

"""
Note: Setting alpha TOO low will result in overfitting and unusable results
"""

lasso0001 = Lasso(alpha=.0001,max_iter=100000).fit(X_train,y_train)
print(f'Train Score: {lasso0001.score(X_train,y_train)}    Test Score: {lasso0001.score(X_test,y_test)}    # of used Features: {np.sum(lasso0001.coef_ != 0)}')
