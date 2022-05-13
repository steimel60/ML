#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn


#------------- Visualize Component Relationships ------------------#
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

#Visualize individual component data
#fig, axes  = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]

#ax = axes.ravel()

#for i in range(30):
#    _, bins = np.histogram(cancer.data[:, i], bins=50)
#    ax[i].hist(malignant[:,i], bins=bins, color=mglearn.cm3(0), alpha=.5)
#    ax[i].hist(benign[:,i], bins=bins, color=mglearn.cm3(2), alpha=.5)
#    ax[i].set_title(cancer.feature_names[i])
#    ax[i].set_yticks(())
#ax[0].set_xlabel("Feature Magnitude")
#ax[0].set_ylabel("Frequency")
#ax[0].legend(["malignant", "benign"], loc='best')
#fig.tight_layout()
#plt.show()

#Use PCA to see relationships between key features
scaler = StandardScaler().fit(cancer.data)  #First we need to scale the data for PCA
X_scaled = scaler.transform(cancer.data)    #Apply the transform calculated in previous line

pca = PCA(n_components=2)                   #Init PCA with top 2 principal components
pca.fit(X_scaled)                           #Fit to breast cancer data

X_pca = pca.transform(X_scaled)             #Transform data on to first 2 principal components
print(f"Original Shape: {X_scaled.shape}    Reduced Shape: {X_pca.shape}")

#Visulaize first 2 components
#plt.figure(figsize=(8,8))
#mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
#plt.legend(cancer.target_names, loc="best")
#plt.gca().set_aspect("equal")
#plt.xlabel("First Principal Component")
#plt.ylabel("Second Principal Component")
#plt.show()


#-------------- Feature Extraction -------------------#
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=.7) #Dataset of celebrity images
image_shape = people.images[0].shape

#fig, axes = plt.subplots(2, 5, figsize=(15,8),subplot_kw={'xticks':(), 'yticks':()})

#for target, image, ax in zip(people.target, people.images, axes.ravel()):
#    ax.imshow(image)
#    ax.set_title(people.target_names[target])
#plt.show()

#Dataset is slightly skewed, many more images of certain people, we will fix that here
#Otherwise feature extraction would be overwhelemed by the likliehood of these people
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

#RGB value to (0,1) scale instead of (0,255) for numeric stability
X_people = X_people/255

#Use knn model to find match for unknown face image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1) #KNN model using the single closest neighbor
knn.fit(X_train,y_train)
print(f'Test Score: {knn.score(X_test,y_test)}')

#Using PCA we hope for better results by learning the principal components
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train) #Note: the whiten paramter is the same as calling StandardScaler
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1) #KNN model using the single closest neighbor
knn.fit(X_train_pca,y_train)
print(f'Test Score: {knn.score(X_test_pca,y_test)}')

#With images we can represent the principal components also as an image
fig, axes = plt.subplots(3,5,figsize=(15,12),subplot_kw={'xticks':(),'yticks':()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),cmap='viridis')
    ax.set_title(f"{i+1}. Component")
plt.show()
