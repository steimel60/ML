#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#----------- decision_function -------------------#
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

X, y = make_circles(noise=.25, factor=.5, random_state=1)

y_named = np.array(['blue','red'])[y] #rename classes

X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train_named)

#decision_function will return float for each n in n_samples
#print(X_test.shape)
#print(gbrt.decision_function(X_test).shape)
#Show first few vals of decision_function
#print(gbrt.decision_function(X_test)[:5]) #positives = 'red' class, negatives = 'blue' class



#-------------- predict_proba --------------------#
"""
Returns probability of each entry being in each given class
"""

#print("Predicted Probabilities:")
#for n in range(5):
#    print(gbrt.predict_proba(X_test)[n])

#---------------- Multiclass Example ---------------#
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=.01, random_state=0).fit(X_train, y_train)
print(gbrt.decision_function(X_test)[:5])
print(gbrt.predict_proba(X_test)[:5])
