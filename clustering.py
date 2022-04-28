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

fig, axes = plt.subplots(1,4,figsize=(15,3),subplot_kw={'xticks':(),'yticks':()})
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),DBSCAN()]

#Random Cluster assignment for reference
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0,high=2,size=len(X))

#Plot random assignment
axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters,cmap=mglearn.cm3, s=60)
axes[0].set_title(f"Random Assignment - ARI: {round(adjusted_rand_score(y,random_clusters),2)}")

#Plot algorithms
for ax, algorithm in zip(axes[1:],algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap=mglearn.cm3, s=60)
    ax.set_title(f"{algorithm.__class__.__name__} - ARI: {round(adjusted_rand_score(y,clusters),2)}")
plt.show()
#ARI is used instead of score because score requires class names to match, but they have no meaning here
