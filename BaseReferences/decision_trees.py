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
#print(tree.feature_importances_)

#---------------- Decision Tree Regressors -------------------#
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

#Get & Visualize data
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
#plt.semilogy(ram_prices.date,ram_prices.price)
#plt.xlabel("Year")
#plt.ylabel("Price in $/Mbyte")
#plt.show()

#Split data by date
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

#Predict prices based on data
X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price) #Log transform for simpler data-target relations
#Show plot above to see why this helps make more linear relation

tree = DecisionTreeRegressor(max_depth=3).fit(X_train,y_train) #init and fit tree model
lr = LinearRegression().fit(X_train, y_train) #lr model for comparison

X_all = ram_prices.date[:,np.newaxis] #predict on all data

pred_tree = tree.predict(X_all)
pred_lr = lr.predict(X_all)

#undo log transform
pred_tree = np.exp(pred_tree)
pred_lr = np.exp(pred_lr)

#Visualize
#plt.semilogy(data_train.date, data_train.price, label = "Training Data")
#plt.semilogy(data_test.date, data_test.price, label = "Testing Data")
#plt.semilogy(ram_prices.date, pred_tree, label = "Tree Prediction")
#plt.semilogy(ram_prices.date, pred_lr, label = "LR Prediction")
#plt.legend()
#plt.show()


#------------------- Random Forests -------------------#
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

"""
Random Forsests generate multiple (different) decision tree models.
It will then decide on an answer based on the returned answers of each model.
"""

X, y = make_moons(n_samples=100, noise=.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2).fit(X_train,y_train)

#Visualize Decision Boundaries
#fig, axes = plt.subplots(2,3,figsize=(20,10))
#for i, (ax,tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#    ax.set_title(f"Tree {i}")
#    mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)

#mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1,-1], alpha=.4)
#axes[-1,-1].set_title("Random Forest")
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.show()

###  Breast Cancer Example  ###
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
forest = RandomForestClassifier(n_estimators=100,random_state=0).fit(X_train,y_train)
print(f"Train Score: {forest.score(X_train,y_train)}    Test Score: {forest.score(X_test,y_test)}")



#----------------------- Gradient Boosted Regression Trees --------------------#
from sklearn.ensemble import GradientBoostingClassifier

"""
Gradient Boosted Classifiers also combine many trees. However, it does so in a serial manner.
Each tree is very simple and the next works to correct the errors of the previous trees.
The parameter learning_rate controls the intensity of the correction.
"""

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)

gbrt = GradientBoostingClassifier(random_state=0).fit(X_train,y_train)
print(f"Train Score: {gbrt.score(X_train,y_train)}    Test Score: {gbrt.score(X_test,y_test)}")
#To Correct overfitting
gbrt_depth = GradientBoostingClassifier(max_depth=1, random_state=0).fit(X_train,y_train)
gbrt_learn = GradientBoostingClassifier(learning_rate=.01, random_state=0).fit(X_train, y_train)
print(f"DEPTH MOD - Train Score: {gbrt_depth.score(X_train,y_train)}    Test Score: {gbrt_depth.score(X_test,y_test)}")
print(f"LEARN RATE MOD - Train Score: {gbrt_learn.score(X_train,y_train)}    Test Score: {gbrt_learn.score(X_test,y_test)}")
