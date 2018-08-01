# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:49:27 2016

@author: vr308
"""

import numpy as np
import matplotlib.pylab as plt
import sklearn.metrics as metrics
import sklearn.tree as tree

from higgs_data import HiggsData
from performance import PerformanceReports

# Single Decision Tree experiments

def performance_summary(trees_,higgs):

    train_error = []
    test_error = []
    roc_curve = []
    berror = []
    y_true_test = higgs.test_true_labels
    y_true_train = higgs.train_true_labels

    for tree_ in trees_:

        y_pred_prob = tree_.predict_proba(higgs.test_scaled)[:,1]
        y_pred_label_test = tree_.predict(higgs.test_scaled)
        y_pred_label_train = tree_.predict(higgs.train_scaled)
        train_error.append(np.subtract(1,metrics.accuracy_score(y_true_train,y_pred_label_train)))
        test_error.append(np.subtract(1,metrics.accuracy_score(y_true_test,y_pred_label_test)))
        roc_curve.append(metrics.roc_curve(y_true_test,y_pred_prob))
        berror.append(PerformanceReports.balanced_classification_error(y_true_test,y_pred_label_test,higgs.test_bweights))

    return train_error, test_error, roc_curve, berror

def fit_tree_depths(higgs,depth=[3,5,7,10,12,15,20,30]):

    trees_ = []
    for i in depth:
        print 'Growing tree of depth ' + str(i)
        dt = tree.DecisionTreeClassifier(min_samples_leaf=200,min_samples_split=150,max_depth=i)
        dt.fit(higgs.train_scaled,higgs.train_true_labels)
        trees_.append(dt)
    return trees_

def fit_tree_max_features(higgs,max_features=['sqrt',10,13,17,21,25,None]):

    trees_ = []
    for i in max_features:
        print 'Growing tree by considering ' + str(i) + ' features at each split'
        dt = tree.DecisionTreeClassifier(max_features=i,min_samples_leaf=200,min_samples_split=150,max_depth=12)
        dt.fit(higgs.train_scaled,higgs.train_true_labels)
        trees_.append(dt)
    return trees_

def fit_tree_min_samples_split(higgs,min_samples_split=[50,100,200,500,1500,3000]):

    trees_ = []
    for i in min_samples_split:
        print 'Trees with minimum splitting requirment of ' + str(i) + ' events'
        dt = tree.DecisionTreeClassifier(max_features=20,min_samples_leaf=200,min_samples_split=i,max_depth=12)
        dt.fit(higgs.train_scaled,higgs.train_true_labels)
        trees_.append(dt)
    return trees_

def fit_tree_min_samples_leaf(higgs,min_samples_leaf=[20,50,100,150,200,300,500]):

    trees_ = []
    for i in min_samples_leaf:
        print 'Trees with minimum requirment of ' + str(i) + ' events at leaves'
        dt = tree.DecisionTreeClassifier(max_features=20,min_samples_leaf=200,min_samples_split=i,max_depth=12)
        dt.fit(higgs.train_scaled,higgs.train_true_labels)
        trees_.append(dt)
    return trees_

if __name__ == "__main__":

    higgs = HiggsData()

    depth_trees_ = fit_tree_depths(higgs)
    max_feature_trees_ = fit_tree_max_features(higgs)
    min_split_trees_ = fit_tree_min_samples_split(higgs)
    min_leaf_trees_ = fit_tree_min_samples_leaf(higgs)

    train_error, test_error, roc_curve, berror = performance_summary(min_leaf_trees_,higgs)