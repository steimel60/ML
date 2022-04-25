#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

"""
Manifold learning is useful for providing visualizations.
However they cannot be applied to datasets they were not trained for so aren't often used as a final product.
They work by making points close to a feature space closer, and moving far points further
"""

#-------- Load Data ------------#
from sklearn. datasets import load_digits

digits = load_digits() #Image dataset of numbers

#----------- Visualize ------------#
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#pca example
pca = PCA(n_components=2).fit(digits.data)
digits_pca = pca.transform(digits.data)
colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
          '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
plt.figure(figsize=(10,10))
plt.xlim(digits_pca[:,0].min(),digits_pca[:,0].max() + 1)
plt.ylim(digits_pca[:,1].min(),digits_pca[:,1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_pca[i,0],digits_pca[i,1],str(digits.target[i]),color=colors[digits.target[i]], fontdict={'weight':'bold', 'size':9})
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

#tSNE example
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10,10))
plt.xlim(digits_tsne[:,0].min(),digits_tsne[:,0].max() + 1)
plt.ylim(digits_tsne[:,1].min(),digits_tsne[:,1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_tsne[i,0],digits_tsne[i,1],str(digits.target[i]),color=colors[digits.target[i]], fontdict={'weight':'bold', 'size':9})
plt.xlabel('tSNE feature 0')
plt.ylabel('tSNE feature 1')
plt.show()
