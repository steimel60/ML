#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn



#----------------- K-Means -------------------#
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans

"""
k-means clustering alternates between 2 steps.
1) Placing a data point in the nearest fitting cluster
2) Adjusting that cluster center to the new mean of the cluster
"""

X,y = make_blobs(random_state=1, n_samples=12) #Load data

kmeans = KMeans(n_clusters=3).fit(X) #Build and fit model
#print(f"Cluster memberships:\n{kmeans.labels_}")
#print(f"Predictions:\n{kmeans.predict(X)}")

#------------ Agglomerative Clustering --------------#
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward

"""
Each point starts as its own cluster.
Clusters merge based on some algorithm.
"""

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

#mglearn.discrete_scatter(X[:,0],X[:,1],assignment)
#plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2'], loc='best')
#plt.xlabel('Feature 0')
#plt.ylabel('Feature 1')
#plt.show()

#visualize process
linkage_arr = ward(X) #Scipy ward returns list of distances bridged during agg clustering
#dendrogram(linkage_arr) #Plot dendrogram
#plt.ylabel("Cluster Distance")
#plt.show()

#------------------ DBSCAN ------------------#
from sklearn.cluster import DBSCAN

"""
Density Based Spatial Clustering of Applications with Noise

Picks arbitrary point, if more than min_samples within distance, eps, then point is a core point.
Core samples closer than eps are put in the same cluster.
"""

dbscan = DBSCAN(min_samples=2, eps=1.5)
clusters = dbscan.fit_predict(X)
#print(f"Cluster Memberships: {clusters}") #-1 is noise
#mglearn.discrete_scatter(X[:,0],X[:,1],clusters)
#plt.legend(['Noise', 'Class 0', 'Class 1'])
#plt.show()


#---------- Compare and Evaluate ----------------#
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=.05, random_state=0) #Complex shapes can be hard for clustering algortihms

scaler = StandardScaler().fit(X) #Zero mean and unit variance
X_scaled = scaler.transform(X)   #Apply transform

#fig, axes = plt.subplots(1,4,figsize=(15,3),subplot_kw={'xticks':(),'yticks':()})
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),DBSCAN()]

#Random Cluster assignment for reference
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0,high=2,size=len(X))

#Plot random assignment
#axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters,cmap=mglearn.cm3, s=60)
#axes[0].set_title(f"Random Assignment - ARI: {round(adjusted_rand_score(y,random_clusters),2)}")

#Plot algorithms
#for ax, algorithm in zip(axes[1:],algorithms):
#    clusters = algorithm.fit_predict(X_scaled)
#    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap=mglearn.cm3, s=60)
#    ax.set_title(f"{algorithm.__class__.__name__} - ARI: {round(adjusted_rand_score(y,clusters),2)}")
#plt.show()
#ARI is used instead of score because score requires class names to match, but they have no meaning here


#-------------- Compare on Face Dataset ----------------------#
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people

"""
Even with robust clustering we have no way of knowing if the clusters are useful without examining them.
For example, we may want to seperate men/women photos but our model learned side vs front view
"""

#Load Celebrity Pics
people = fetch_lfw_people(min_faces_per_person=20, resize=.7) #Dataset of celebrity images
image_shape = people.images[0].shape

#Dataset is slightly skewed, many more images of certain people, we will fix that here
#Otherwise feature extraction would be overwhelemed by the likliehood of these people
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

#RGB value to (0,1) scale instead of (0,255) for numeric stability
X_people = X_people/255

#PCA gives us better features than the raw data
pca = PCA(n_components=100, whiten=True, random_state=0)
X_pca = pca.fit_transform(X_people)

#DBSCAN analysis
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)

#for cluster in range(max(labels)+1):
#    mask = labels == cluster
#    n_images = np.sum(mask)
#    fig, axes = plt.subplots(1, n_images, figsize=(n_images*1.5, 4), subplot_kw={'xticks':(),'yticks':()})
#    for image, label, ax in zip(X_people[mask], y_people[mask], axes):
#        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#        ax.set_title(people.target_names[label].split()[-1])
#plt.show()



#KMeans analysis
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)

#Plot cluster centers
fig, axes = plt.subplots(2, 5, subplot_kw={'xticks':(),'yticks':()}, figsize=(12,4))
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
plt.show()



#Agglomerative Clustering analysis
agg = AgglomerativeClustering(n_clusters=40)
labels_agg = agg.fit_predict(X_pca)

for cluster in [10, 13, 19, 22, 36]: #Hand picked "interesting" clusters
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 10, subplot_kw={'xticks':(),'yticks':()}, figsize=(15,8))
    axes[0].set_ylabel(np.sum(mask))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label][-1], fontdict={'fontsize': 9})
plt.show()
