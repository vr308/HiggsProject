# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:44:23 2016

@author: vr308
"""

import os
import sys
import pandas as pd
import warnings
import matplotlib.pylab as plt
import numpy as np
sys.path.append(os.path.join(os.path.realpath('..')))

def plot_depth_estimators_heatmap(grid_rf, grid_et, max_feature):
        sub_rf = grid_rf[grid_rf.max_features == max_feature][['test_error']]
        sub_et = grid_et[grid_et.max_features == max_feature][['test_error']]
        n_trees = len(np.unique(grid_rf.n_estimators))
        max_depth = len(np.unique(grid_rf.max_depth))
        heat_rf = np.array(sub_rf).reshape(n_trees, max_depth)
        heat_et = np.array(sub_et).reshape(n_trees, max_depth)
        plt.figure(figsize=(10,8))
        plt.subplot(1,2,1)
        plt.imshow(heat_rf, aspect='auto')
        plt.xticks(xrange(max_depth), np.unique(grid_rf.max_depth.values))
        plt.yticks(xrange(n_trees), np.unique(grid_rf.n_estimators.values))
        plt.xlabel('Max depth', fontsize='small')
        plt.ylabel('N. of trees', fontsize='small')
        plt.title('Random Forest', fontsize='small')
        plt.subplot(1,2,2)
        plt.imshow(heat_et, aspect='auto')
        plt.colorbar(shrink=0.9, format='%.4f')
        plt.xlabel('Max depth', fontsize='small')
        plt.ylabel('N. of trees', fontsize='small')
        plt.xticks(xrange(max_depth), np.unique(grid_rf.max_depth.values))
        plt.yticks(xrange(n_trees), np.unique(grid_rf.n_estimators.values))
        plt.title('Extremely Random trees', fontsize='small')
        plt.suptitle('Balanced classification error')
        plt.savefig('../Graphs/Grid_error_depth_trees_rf_et.png')


def plot_depth_feature_heatmap(grid_rf, grid_et, n_trees):
        sub_rf = grid_rf[grid_rf.n_estimators == n_trees][['test_error']]
        sub_et = grid_et[grid_et.n_estimators == n_trees][['test_error']]
        max_features = len(np.unique(grid_rf.max_features))
        max_depth = len(np.unique(grid_rf.max_depth))
        heat_rf = np.array(sub_rf).reshape(max_depth, max_features)
        heat_et = np.array(sub_et).reshape(max_depth, max_features)
        plt.figure(figsize=(10,8))
        plt.subplot(1,2,1)
        plt.imshow(heat_rf, aspect='auto', interpolation=None)
        plt.yticks(xrange(max_depth), np.unique(grid_rf.max_depth.values))
        plt.xticks(xrange(max_features), np.unique(grid_rf.max_features.values))
        plt.ylabel('Max depth', fontsize ='small')
        plt.xlabel('Max features', fontsize='small')
        plt.title('Random Forest', fontsize='small')
        plt.subplot(1,2,2)
        plt.imshow(heat_et, aspect='auto')
        plt.colorbar(shrink=0.9, format='%.4f')
        plt.ylabel('Max depth', fontsize='small')
        plt.xlabel('Max features', fontsize='small')
        plt.yticks(xrange(max_depth), np.unique(grid_rf.max_depth.values))
        plt.xticks(xrange(max_features), np.unique(grid_rf.max_features.values))
        plt.title('Extremely Random trees', fontsize='small')
        plt.suptitle('Balanced classification error')
        plt.savefig('../Graphs/Grid_error_depth_features_rf_et.png')


if __name__ == "__main__":


    print 'Reading grid scores of RF and ET models'

    grid_rf = pd.read_csv('../Results/grid_scores_RF_weighted.csv')
    grid_et = pd.read_csv('../Results/grid_scores_ET_weighted.csv')

    warnings.filterwarnings('ignore')
    plot_depth_estimators_heatmap(grid_rf, grid_et, max_feature=25)
    plot_depth_feature_heatmap(grid_rf, grid_et, n_trees=100)

