#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn


#------------- Linear SVC ------------------#
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X,y = make_blobs(centers=4, random_state=8)
y = y % 2

#mglearn.discrete_scatter(X[:,0],X[:,1],y) #Features are the axes, y gives its class
#plt.show()

linear_svm = LinearSVC().fit(X,y)

#mglearn.plots.plot_2d_separator(linear_svm, X) #linear model is bad for this dataset
#plt.show()

#add feature1**2 as a new feature
X_new = np.hstack([X, X[:,1:]**2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
#fig = plt.figure()
#Visualize in 3D
#ax = Axes3D(fig, elev=-152, azim=-26)
#plot classes separately
mask = y==0
#ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
#ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
#ax.set_xlabel('Feature 1')
#ax.set_ylabel('Feature 2')
#ax.set_zlabel('Feature 3')
#plt.show()

"""
We can now separate the 2 classes linearly with a plane in 3D
"""

linear_svm_3d = LinearSVC().fit(X_new,y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

#Visualize
#fig = plt.figure()
#ax = Axes3D(fig, elev=-152, azim=-26)
xx = np.linspace(X_new[:,0].min() - 2, X_new[:,0].max() + 2, 50)
yy = np.linspace(X_new[:,1].min() - 2, X_new[:,1].max() + 2, 50)

XX, YY = np.meshgrid(xx,yy)
ZZ = (coef[0]*XX + coef[1]*YY + intercept) / -coef[2]
#ax.plot_surface(XX,YY,ZZ,rstride=8,cstride=8,alpha=.3)
#plot classes separately
mask = y==0
#ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
#ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
#ax.set_xlabel('Feature 1')
#ax.set_ylabel('Feature 2')
#ax.set_zlabel('Feature 3')
#plt.show()


#--------------- Visualizing how SVMs work ----------------------#
from sklearn.svm import SVC

X,y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)

#plot support vectors
sv = svm.support_vectors_
#class labels are given the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
#mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.show()

#Visualize Parameter Settings
#fig, axes = plt.subplots(3,3, figsize=(15,10))

#for ax, C in zip(axes, [-1, 0, 3]):
#    for a, gamma in zip(ax, range(-1,2)):
#        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
#axes[0,0].legend(['class 0', 'class 1', 'class 2', 'sv class 0', 'sv calss 1'], ncol=4,loc=(.9,1.2))
#plt.show()


#---------------- Breast Cancer Example ------------------------#
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svc = SVC(C=1000).fit(X_train,y_train)
print(f"Train Score: {svc.score(X_train,y_train)}    Test Score: {svc.score(X_test,y_test)}")
