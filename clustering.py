#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn



#----------------- K-Means -------------------#
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

"""
k-means clustering alternates between 2 steps.
1) Placing a data point in the nearest fitting cluster
2) Adjusting that cluster center to the new mean of the cluster
"""

X,y = make_blobs(random_state=1) #Load data

kmeans = KMeans(n_clusters=3).fit(X) #Build and fit model
print(f"Cluster memberships:\n{kmeans.labels_}")
print(f"Predictions:\n{kmeans.predict(X)}")
