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



#------------------------- Grid Search --------------------------------------#
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

"""
Grid Search loops over possible parameter options to find the best combination.
For example, consider the C and gamma parameters in the SVC class.
If we want to check C=.001 and .1 and gamma = .001 and 100 we get the following grid
                C=.001                      C=.1
gamma=.001      SVC(C=.001, gamma=.001)     SVC(C=.1, gamma=.001)
gamma=100       SVC(C=.001, gamma=100)      SVC(C=.1, gamma=100)
"""

#If we use our test set to adjust parameters we can no longer trust that it's not overfitting
#We can test on a "validation set" to fix this
mglearn.plots.plot_threefold_split()
#plt.show()

#Split training and validation data from our test set
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
#Further split our training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=1)

#Grid search
best_score=0
for gamma in [.001, .01, .1, 1, 10, 100]:
    for C in [.001, .01, .1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}
svm = SVC(**best_parameters).fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print(f"Best score on Validation set: {best_score}")
print(f"Best parameters: {best_parameters}")
print(f"Test Score with best params: {test_score}")

"""
We can get a better idea of the generalization performance by combining grid search and cross validation.
"""
best_score=0
for gamma in [.001, .01, .1, 1, 10, 100]:
    for C in [.001, .01, .1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}
svm = SVC(**best_parameters).fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print(f"Best score on Validation set: {best_score}")
print(f"Best parameters: {best_parameters}")
print(f"Test Score with best params: {test_score}")


"""
Because this is common practice sklearn has a class to help with the process
"""
param_grid = {'C':[.001, .01, .1, 1, 10, 100],'gamma':[.001, .01, .1, 1, 10, 100]} #Dict with parameters we want to check
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)
print(f"GridSearchCV Score: {grid_search.score(X_test, y_test)}")
print(f"GSCV Best Params: {grid_search.best_params_}")
print(f"GSCV Best Score: {grid_search.best_score_}")
print(f"GSCV Best Estimator: {grid_search.best_estimator_}")

#It can be useful to see the results of each iteration
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

#Nested cross validation can be useful to determine how well a model works on a particular dataset
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)
print(f"Cross Val Scores: {scores}    Mean: {np.mean(scores)}")


#------------------ Imbalanced Data Sets ----------------------$
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train) #Always predict "not 9"
pred_most_freq = dummy_majority.predict(X_test)
print(f"Unique predicted labels: {np.unique(pred_most_freq)}")
print(f"Test Score: {dummy_majority.score(X_test, y_test)}")

tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print(f"Tree Score: {tree.score(X_test, y_test)}")

#Because data imbalanced our score metric is not a good indicator
#use confusion_matrix to see type 1 and 2 errors
confusion = confusion_matrix(y_test, pred_tree)
print(confusion)
