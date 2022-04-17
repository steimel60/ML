#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

"""
Similar to linear regression models except we also have class thresholds, resulting in the formula:
y = w[0]*x[0] + w[1]*x[1] + ... + w[n]*x[n] + b > 0.
Where the > 0 tells us if the resulting class is in the -1 or 1 class.
"""

#----------- Visualize Simple Model ---------------#
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

#fig, axes = plt.subplots(1,2, figsize=(10,3))

#for model, ax in zip([LinearSVC(), LogisticRegression()], axes): #2 plots with the 2 different models
#    clf = model.fit(X,y)
#    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=.5, ax=ax, alpha=.7) #Separate graph
#    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax) #plot points
#    ax.set_title(clf.__class__.__name__)
#    ax.set_xlabel("Feature 0")
#    ax.set_ylabel("Feature 1")
#axes[0].legend()
#plt.show()
#mglearn.plots.plot_linear_svc_regularization() #Show how the C parameter affects the model
#plt.show()

#---------------- Real World Example -----------------#
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer() #Load data
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42) #split data

logreg = LogisticRegression().fit(X_train, y_train) #Build logistiv regression model
print(f'Train Score: {logreg.score(X_train, y_train)}    Test Score: {logreg.score(X_test, y_test)}')

"""
With our traing and testing scores so close, we are likely underfitting despite good performance.
"""

logreg1 = LogisticRegression(C=100).fit(X_train, y_train)
print(f'Train1 Score: {logreg1.score(X_train, y_train)}    Test1 Score: {logreg1.score(X_test,y_test)}')

#Plot coefficeints to see relevancy
#plt.plot(logreg.coef_.T, 'o', label='C=1')
#plt.plot(logreg1.coef_.T, '^', label='C=100')
#plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
#plt.hlines(0,0,cancer.data.shape[1])
#plt.ylim(-5,5)
#plt.xlabel("Feature")
#plt.ylabel("Coefficeint Magnitude")
#plt.legend()
#plt.show()

#-------------- 1 vs the Rest ----------------#
from sklearn.datasets import make_blobs

"""
Many linear classification models are binary and only work for 2 classes.
To get around this we can by running model for each class and determining "in or out"
based on the confidence levels of each class for each point.
"""

#Get and visualize dataset
X,y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.legend(["Class 0", "Class 1", "Class 2"])
#plt.show()

#Train LinearSVC Classifier on dataset
linear_svm = LinearSVC().fit(X,y)
print(f"Coef Shape: {linear_svm.coef_.shape}")
print(f"Intercept Shape: {linear_svm.intercept_.shape}")
mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)

for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Line Class 0", "Line Class 1", "Line Class 2"], loc=(1.01,.3))
plt.show()
