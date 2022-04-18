#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import graphviz

#--------------- quick visual ----------------#

#mglearn.plots.plot_animal_tree()
#plt.show()

#-------------- Decision Tree Classifier ------------------#
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print(f"Train Score: {tree.score(X_train,y_train)}    Test Score: {tree.score(X_test,y_test)}")
#Without pruning we overfit our training set, memorizing every leaf

tree1 = DecisionTreeClassifier(max_depth=4,random_state=0).fit(X_train, y_train) #max depth of 4 limits us to 4 "questions" when building the model
#So we don't go down to every individual leaf
print(f"Train1 Score: {tree1.score(X_train,y_train)}    Test1 Score: {tree1.score(X_test,y_test)}")
print(tree.feature_importances_)
