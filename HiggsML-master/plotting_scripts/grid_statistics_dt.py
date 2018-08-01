# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:39:22 2016

@author: vr308
"""
import warnings
import matplotlib.pylab as plt
import scipy.interpolate as intp
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.realpath('..')))

def plot_evolution_depth(max_features, min_leaf, min_split):
    sub_ent = dt_weighted[(dt_weighted.max_features == 10) & (dt_weighted.criterion == 'entropy') & (dt_weighted.min_samples_leaf == min_leaf) & (dt_weighted.min_samples_split == min_split)]
    sub_gini = dt_weighted[(dt_weighted.max_features == 10) & (dt_weighted.criterion == 'gini') & (dt_weighted.min_samples_leaf == min_leaf) & (dt_weighted.min_samples_split == min_split)]
    max_depth = dt_weighted['max_depth'].unique()
    xnew = np.linspace(max_depth.min(), max_depth.max(), 20)
    f_ent = intp.interp1d(max_depth, sub_ent.test_error.values, kind='quadratic')
    f_gini = intp.interp1d(max_depth, sub_gini.test_error.values, kind='quadratic')
    plt.figure()
    plt.plot(xnew, f_ent(xnew), label = 'Entropy')
    plt.plot(xnew, f_gini(xnew), label = 'Gini')
    plt.xlabel('Max Depth')
    plt.ylabel('%')
    plt.title('Balanced Classification Error')
    plt.legend()
    plt.savefig('../Graphs/tree_depth_evolution.png')


def plot_evolution_max_features(max_depth, min_leaf, min_split):
    sub_ent = dt_weighted[(dt_weighted.max_depth == max_depth) & (dt_weighted.criterion == 'entropy') & (dt_weighted.min_samples_leaf == min_leaf) & (dt_weighted.min_samples_split == min_split)]
    sub_gini = dt_weighted[(dt_weighted.max_depth == max_depth) & (dt_weighted.criterion == 'gini') & (dt_weighted.min_samples_leaf == min_leaf) & (dt_weighted.min_samples_split == min_split)]
    maxf = dt_weighted['max_features'].unique()
    xnew = np.linspace(maxf.min(), maxf.max(), 20)
    f_ent = intp.interp1d(maxf, sub_ent.test_error.values, kind='quadratic')
    f_gini = intp.interp1d(maxf, sub_gini.test_error.values, kind='quadratic')
    plt.figure()
    plt.plot(xnew, f_ent(xnew), label = 'Entropy')
    plt.plot(xnew, f_gini(xnew), label = 'Gini')
    plt.xlabel('Max Features')
    plt.ylabel('%')
    plt.title('Balanced Classification Error')
    plt.legend()
    plt.savefig('../Graphs/tree_feature_evolution.png')


def plot_depth_feature_heatmap(dt_weighted):
    pruning_parameter = np.unique(dt_weighted.min_samples_leaf)
    plt.figure(figsize=(8,10))
    for k, i in zip(pruning_parameter, [(1,2),(3,4),(5,6)]):
        sub = dt_weighted[(dt_weighted.min_samples_leaf == k) & (dt_weighted.min_samples_split == k)]
        scores = sub[['max_depth','max_features','test_error', 'criterion']]
        gini = scores[scores.criterion == 'gini']
        ent = scores[scores.criterion == 'entropy']
        max_depth = len(np.unique(scores.max_depth))
        max_features = len(np.unique(scores.max_features))
        heat_gini = np.array(gini.test_error).reshape(max_depth, max_features)
        heat_ent = np.array(ent.test_error).reshape(max_depth, max_features)
        plt.subplot(str(32) + str(i[0]))
        plt.grid()
        plt.imshow(heat_gini, interpolation='gaussian', cmap=plt.cm.jet,vmin=min(dt_weighted.test_error), vmax= max(dt_weighted.test_error))
        plt.yticks(xrange(max_depth), np.unique(gini.max_depth.values))
        plt.xticks(xrange(max_features), np.unique(gini.max_features.values))
        plt.xlabel('Max Features', fontsize='small')
        plt.ylabel('Max depth', fontsize='small')
        plt.title('Gini', fontsize='small')
        plt.subplot(str(32) + str(i[1]))
        plt.title('Entropy', fontsize='small')
        plt.grid()
        plt.imshow(heat_ent, interpolation='gaussian', cmap=plt.cm.jet, vmin=min(dt_weighted.test_error), vmax= max(dt_weighted.test_error))
        plt.yticks(xrange(max_depth), np.unique(gini.max_depth.values))
        plt.xticks(xrange(max_features), np.unique(gini.max_features.values))
        plt.xlabel('Max Features', fontsize='small')
        plt.ylabel('Max depth', fontsize='small')
        plt.colorbar(shrink=0.7)
        plt.subplots_adjust(hspace=0.3)
    plt.suptitle('Pruning Parameters: ' + str(list(pruning_parameter.values)))
    plt.savefig('../Graphs/Grid_error_gini_entropy.png')

if __name__ == "__main__":

    print 'Reading saved results for DT'

    dt_weighted = pd.read_csv('../Results/grid_scores_DT_weighted.csv')

    print 'Saving plots..'

    warnings.filterwarnings('ignore')
    plot_depth_feature_heatmap(dt_weighted)
    plot_evolution_depth(max_features=25, min_leaf=100, min_split=100)
    plot_evolution_max_features(max_depth=15, min_leaf=100, min_split=100)

    print 'Done'



