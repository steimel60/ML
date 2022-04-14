#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#----------------- Load and view data -----------------------#
from sklearn.datasets import load_iris

iris_dataset = load_iris() #load in our data
#print(iris_dataset.keys()) #returns a "bunch" object, similar to dict
#print(iris_dataset['DESCR']) #Description
#print(iris_dataset['data']) #flower data in numpy array

#---------------- Create train/test split --------------------#
from sklearn.model_selection import train_test_split

#Split data set into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#print("X train shape:", X_train.shape)
#print("y train shape:", y_train.shape)

#print("X test shape:",X_test.shape)
#print("y test shape:",y_test.shape)

#------------------ Visualize the Data ------------------------#

df = pd.DataFrame(X_train, columns=iris_dataset.feature_names) #create pandas df
#Plot a "pair plot", color by y_train value. Shows how features relate - limited to 2 at a time
pd.plotting.scatter_matrix(df, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20},s=60,alpha=.8)
#plt.show()

#---------------- Create KNeighbors Model ----------------------#
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1) #model instance using single closest neighbor
knn.fit(X_train, y_train) #build model using train data

#--------------------- Make Predictions ------------------------#

X_new = np.array([[5, 2.9, 1, .2]]) #Array representing a flower we want to classify
prediction = knn.predict(X_new) #Our model's prediction
print(prediction) #Our 'target' col is represented by ints
print(iris_dataset['target_names'][prediction]) #corresponding name

#--------------------- Evaluate Model ---------------------------#

y_pred = knn.predict(X_test)    #Get predictions on our testing set
acc = np.mean(y_pred==y_test)   #Compare them to known results
print(acc)
print(knn.score(X_test,y_test)) #Another way to do the same thing
