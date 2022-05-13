#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

"""
Non-Negative Matrix Factorization (NMF) can be used to extract useful features.
It's very useful with data thats additive - eg) Music with lots of instruments, a recording with many voices etc.
"""

#-------------- Load Data -------------------#
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

#--------------------- NMF ------------------------#
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
#Transform data into n_components
nmf = NMF(n_components=15, random_state=0).fit(X_train) #Calculate transform
X_train_nmf = nmf.transform(X_train)                    #Transform X_train
X_test_nmf = nmf.transform(X_test)                      #Transform X_test

#plot data
#fig, axes = plt.subplots(3,5,figsize=(15,12),subplot_kw={'xticks':(),'yticks':()})
#for i, (component, ax) in enumerate(zip(nmf.components_,axes.ravel())):
#    ax.imshow(component.reshape(image_shape))
#    ax.set_title(f"{i}. Component")
#plt.show()


#------------ Visualizing Decomposition of Additive Data ------------#
S = mglearn.datasets.make_signals() #load additive data

#plot dataset
plt.figure(figsize=(6,1))
plt.plot(S,'-')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()

#Assume 100 ways to record/measure the data
A = np.random.RandomState(0).uniform(size=(100,3)) #100 devices measuring the 3 channels
X = np.dot(S, A.T)  #Signal sent to the 3 channels, mixed with strength/noise generated line above

nmf = NMF(n_components=3, random_state=42) #init NMF
S_ = nmf.fit_transform(X) #calculate and apply nmf transform

pca = PCA(n_components=3) #init PCA for comparison
H = pca.fit_transform(X)  #calculate and apply pca transform

#compare data on plots
models = [X, S, S_, H]
names = ['Observations (first 3 measurements)', 'True Sources', 'NMF recovered signals', 'PCA recovered signals']

fig, axes = plt.subplots(4, figsize=(8,4), gridspec_kw={'hspace':.5}, subplot_kw={'xticks':(),'yticks':()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:,:3],'-')
plt.show()
