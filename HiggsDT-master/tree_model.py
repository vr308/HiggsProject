# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 03:21:29 2016

@author: vr308
"""

from sklearn.tree import tree
import numpy as np

def fit_tree(X_train, Y_train,W_train,criterion, max_depth,min_samples_leaf,min_samples_split,max_features):
    
    model = tree.DecisionTreeClassifier(criterion=criterion,
                                        class_weight='auto',
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split= min_samples_split,
                                        max_features=max_features)
    model.fit(X_train,Y_train,W_train)
    return model
    

def balanced_classification_error(model,X,Y,W):
    
        Y_pred = model.predict(X)
        Y_miss = (Y != Y_pred)
        return sum(W*Y_miss)
    
def model_probability(tree_estimator,X,y):
    
    return tree_estimator.predict_proba(X)[:,1]    
    
    
def calibrate_probability(tree_estimator,X,y):
    
    X = np.asarray(X).astype(np.float32)
    
    tree_value = tree_estimator.tree_.value
    leaf_index = tree_estimator.tree_.apply(X)
    class_split_in_leaf = tree_value[leaf_index][:,0]
    all_samples_in_leaf = class_split_in_leaf.sum(axis=1)
    
    #signals = sum(1 for x in y if x == 1)
    #background = sum(1 for x in y if x == 0)
        
    #base_rate = np.float32(signals)/np.float32(background)
    base_rate = 0.3
    
    m = 10/base_rate
    
    nr = (class_split_in_leaf[:,1] + base_rate*m)
    dr = (all_samples_in_leaf + m)
    
    calibrated_prob = nr/dr
    
    return calibrated_prob 