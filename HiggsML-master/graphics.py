# -*- coding: utf-8 -*-
"""
@author: vr308

This script contains plotting commands for visualizing :

    - Univariate Feature distributions (background + signal)
    - Bi-variate Feature distributions (background + signal)
    - Correlation heatmap of Derived features, Primary features and all features
    - Correlation co efficients Features vs. Class
"""

import sys
sys.path.append('plotting_scripts/')

import hexbin
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

import ConfigParser
settings = ConfigParser.ConfigParser()
settings.read('settings.ini')


class Graphics:

    @classmethod
    def plot_hist(cls, b, s, feature):

        plt.figure()
        plt.grid()
        plt.hist(b[feature], bins=200, color='b', alpha=0.5, label='b', weights=b.Weight, normed=True)
        plt.hist(s[feature], bins=200, color='r', alpha=0.3, label='s', weights=s.Weight, normed=True)
        plt.title(feature)
        plt.legend()
        plt.savefig('Graphs/' + feature + '_Hist.png', bbox_inches='tight')

    @classmethod
    def plot_feature_corr_matrix(cls, df, title, figsave):

        corr = df.corr()
        xmax = df.shape[1] + 0.5
        save_index = df.columns[0].split('_')[0]
        plt.figure(figsize=(14, 11))
        plt.pcolor(corr, cmap='rainbow')
        plt.colorbar()
        plt.xticks(np.arange(0.5, xmax), corr.columns, rotation=90, fontsize=8)
        plt.yticks(np.arange(0.5, xmax), corr.columns, fontsize=8)
        plt.title(title)
        plt.xlim(0, xmax - 0.5)
        plt.ylim(0, xmax - 0.5)
        plt.savefig('Graphs/' + save_index + 'FeaturesHeatMap.png', bbox_inches='tight')

    @classmethod
    def plot_bi_features(cls, b, s, f1, f2):

        plt.figure()
        plt.grid()
        plt.plot(b[f1], b[f2], 'bo', label='b', markersize=0.5)
        plt.plot(s[f1], s[f2], 'r+', label='s', markersize=0.5, alpha=0.3)
        plt.legend()
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f1 + ' vs. ' + f2, fontsize='small')
        plt.savefig('Graphs/' + f1 + '_vs_' + f2 + '.png', bbox_inches='tight')

    @classmethod
    def plot_roc_auc(cls, fpr, tpr, score, label):

        plt.plot(fpr, tpr, label=label + '=' + str(score))
        plt.xlabel('(1-Specificity) or False Positive Rate')
        plt.ylabel('Sensitivity or True Positive Rate')
        plt.grid()
        plt.legend(loc=4)



#if __name__ == "__main__":
#
#    #Loading the dataset
#    from higgs_data import HiggsData
#
#    hd = HiggsData()
#
#    # Get background/signal samples
#    b = hd.background
#    s = hd.signal
#
#    features = hd.train.columns
#
#    f1 = 'DER_mass_transverse_met_lep' # Sample feature 1
#    f2 = 'DER_mass_MMC' # Sample feature 2
#
#    # Feature Heat Map
#    title =  'Feature Correlation Heatmap : Primary & Derived Features'
#    figsave=True
#
#    Graphics.plot_feature_corr_matrix(hd.train,title,figsave)
#
#    # Feature-Class Heat Map
#    Graphics.plot_feature_class_corr_matrix(hd.train,hd.train_true_labels,features)
#
#    # Bi Feature Distribution
#    Graphics.plot_bi_features(b,s,f1,f2)
#
#    f1 = 'DER_mass_transverse_met_lep'
#
#    # Histogram of univariate feature
#    Graphics.plot_hist(b,s,f1,isNorm=True)
