# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:58:36 2016

@author: vr308
"""

import os

os.chdir(os.getcwd()+ '/higgsSVM')

import numpy as np
import sys
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pylab as plt
import itertools

import preprocessing 
import cross_validation 
import discovery_significance


df = preprocessing.load_data(path='/local/data/public/vr308/')  
df = preprocessing.drop_missing_values(df)
df_normed = preprocessing.normalize(df)[0]
df_features = preprocessing.get_features(df)

# Get background/signal samples
b = df[df.Label == 1]
s = df[df.Label == 0]
    
X = np.asarray(df[['DER_mass_MMC','A']])[0:5000]
Y = np.asarray(df.Label[0:5000])

C = 100
gamma = 0.005
 
clf = cross_validation.fit_svm(X,Y,'rbf',C,gamma)

h=1

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
                
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure()
plt.grid()
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('DER_mass_MMC')
plt.ylabel('A')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Gaussian SVM Decision Surface')
plt.xticks(())
plt.yticks(())

# Metrics

Y_true = Y
Y_pred = clf.predict(X)
print(metrics.classification_report(Y_true,Y_pred))

#plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],marker='+',color='r')