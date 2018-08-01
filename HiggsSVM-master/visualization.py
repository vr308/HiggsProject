# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:43:52 2016

@author: vr308

This script contains plotting commands for visualizing :

    - Univariate Feature distributions (only, only signal, background + signal)
    - Bi-variate Feature distributions (only, only signal, background + signal)
    - Correlation heatmap of Derived features, Primary features and all features
    - Correlation co efficients Features vs. Class
"""

import matplotlib.pyplot as plt
import scipy.stats as stats
import preprocessing
import numpy as np


def plot_hist(b,s,feature,isNorm):

    plt.figure()
    plt.grid()
    plt.hist(b[feature], bins=200, color='b', alpha=0.5,label = 'b', normed=isNorm)
    plt.hist(s[feature], bins=200, color='r', alpha=0.3,label = 's', normed=isNorm)
    plt.title(feature)
    plt.legend()
    plt.savefig('Graphs/' + feature + '_Hist.png',bbox_inches='tight')


def plot_feature_corr_matrix(df,title,figsave):

    corr = df.corr()
    xmax = df.shape[1] + 0.5
    save_index = df.columns[0].split('_')[0]

    plt.figure(figsize=(14,11))
    plt.pcolor(corr,cmap='rainbow')
    plt.colorbar()
    plt.xticks(np.arange(0.5,xmax),corr.columns, rotation=90, fontsize=8)
    plt.yticks(np.arange(0.5,xmax),corr.columns,fontsize=8)
    plt.title(title)
    plt.xlim(0,xmax-0.5)
    plt.ylim(0,xmax-0.5)
    plt.savefig('Graphs/' + save_index + 'FeaturesHeatMap.png',bbox_inches='tight')


def plot_feature_class_corr_matrix(df,labels,cols):

    corr = []
    for i in xrange(0,df.shape[1]):
        corr.append(stats.pointbiserialr(labels,df.icol(i)))
    pos = np.arange(1,df.shape[1] +1)
    c, p = zip(*corr)
    plt.figure(figsize=(10,10))
    plt.grid()
    plt.stem(pos,c)
    plt.xticks(np.arange(1,max(pos)+1),cols,rotation=90,fontsize=8)
    plt.xlim(xmax=max(pos)+1)
    plt.ylabel('Correlation')
    plt.title('Point biserial Correlation - Features v. Class')
    plt.savefig('Graphs/BiSerialCorr.png',bbox_inches='tight')


def plot_bi_features(b,s,f1,f2):

    plt.figure()
    plt.plot(b[f1],b[f2],'bo', label='b',markersize=0.5)
    plt.plot(s[f1],s[f2], 'r+',label='s',markersize=0.5)
    plt.legend()
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f1 + ' vs. ' + f2)
    plt.savefig('Graphs/' + f1 + '_vs_' + f2+ '.png',bbox_inches='tight')

def plot_miss_samples(cls,b,s,b_z,s_z,f1,f2):

    plt.subplot(121)
    plt.grid()
    plt.plot(b[f1],b[f2],'bo', label='b',markersize=0.7)
    plt.plot(s[f1],s[f2], 'r+',label='s',markersize=0.7)
    plt.legend(loc=2)
    plt.xlim(b[f1].mean() - 2*b[f1].std(),b[f1].mean() + 2*b[f1].std())
    plt.ylim(b[f2].mean() - 1*b[f2].std(),b[f2].mean() + 2*b[f2].std())
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title('Classifier Hits')
    plt.subplot(122)
    plt.grid()
    plt.plot(s_z[f1],s_z[f2],'m*',label='missed s',markersize=4)
    plt.plot(b_z[f1],b_z[f2],'c*',label='missed b',markersize=4)
    plt.legend(loc=2)
    plt.xlim(b[f1].mean() - 2*b[f1].std(),b[f1].mean() + 2*b[f1].std())
    plt.ylim(b[f2].mean() - 1*b[f2].std(),b[f2].mean() + 2*b[f2].std())
    plt.title('Classifier Misses')
    plt.savefig('Graphs/' + f1 + '_vs_' + f2+ '.png',bbox_inches='tight')


def plot_grid_scores(cls,scores, C_range,gamma_range,metric, train_sample_type,cmap,vmin,vmax,grid_density):

    # Plot heatmap of performance metrics as a function of gamma and C

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores,vmin=vmin,vmax=vmax,interpolation='nearest', aspect='auto',cmap=cmap)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), np.round(gamma_range,7), rotation=45,fontsize='small')
    plt.yticks(np.arange(len(C_range)), np.round(C_range,4), fontsize='small')
    plt.title(metric + ' metric ' + ' [ Training Set : ' + train_sample_type + ' ]')
    plt.savefig('Graphs/' + metric + '_' + train_sample_type + '_grid'+ str(grid_density) +'.png',bbox_inches='tight')

def plot_ams_curve(thresh,ams_curve,label,train_sample_type):

    plt.figure()
    plt.grid()
    plt.stem(thresh,ams_curve,linefmt='b-')
    plt.plot(thresh, ams_curve,'bo')
    plt.xlabel('Threshold %')
    plt.ylabel('AMS Test')
    plt.title('AMS computed on test data')
    plt.xlim(80,90)
    plt.savefig('Graphs/AMS_Optimum_' + train_sample_type + '.png',bbox_inches='tight')

if __name__ == "__main__":

    # Loading the dataset
    df = preprocessing.load_data(path='/home/raid/vr308/workspace/Python/higgsSVM/Data/')
    df = preprocessing.drop_missing_values(df)
    df_normed = preprocessing.normalize(df)[0]
    df_features = preprocessing.get_features(df)

   # Get background/signal samples
    b = df[df.Label == 1]
    s = df[df.Label == 0]

    features = df_features.columns

    f1 = 'A' # Sample feature 1
    f2 = 'DER_mass_MMC' # Sample feature 2

    # Feature Heat Map
    title =  'Feature Correlation Heatmap : Primary & Derived Features'
    figsave=True

    plot_feature_corr_matrix(df_features,title,figsave)

    #Feature-Class Heat Map
    plot_feature_class_corr_matrix(preprocessing.get_features(df),df.Label,features)

    # Bi Feature Distribution
    plot_bi_features(b,s,f1,f2)

    f1 = 'A'
    # Histogram of univariate feature
    plot_hist(b,s,f1,isNorm=True)








