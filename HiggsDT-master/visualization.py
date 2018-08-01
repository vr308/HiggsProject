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

import matplotlib.pylab as plt
import scipy.stats as stats
import preprocessing
import numpy as np


def plot_hist(b,s,feature,isNorm):
    
    plt.figure()
    plt.grid()
    plt.hist(b[feature], bins=200, color='b', alpha=0.5,label = 'b', normed=isNorm)
    plt.hist(s[feature], bins=200, color='r', alpha=0.3,label = 's', normed=isNorm)
    plt.title('Distribution of Feature :' + feature)
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel(feature)
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
    plt.xticks(pos,cols,rotation=90,fontsize=8)
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
    
def plot_prob_dist(model,X_train,X_test):
    
    y_prob_train = model.predict_proba(X_train)
    y_prob_test = model.predict_proba(X_test)
    
    plt.figure()
    plt.hist(y_prob_train, bins=100)
    plt.title('Prediction Scores for Training set')
    plt.figure()
    plt.hist(y_prob_test,bins=100)
    plt.title('Prediction Scores for Test set')

def plot_grid_scores(scores,metric,split,max_depth_range, min_samples_leaf_range):
        
    plt.figure(figsize=(10,7))   
    plt.subplots_adjust(left=.2, right=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.jet)
    plt.ylabel('max_depth')
    plt.xlabel('min_samples_leaf')
    plt.colorbar()
    plt.yticks(np.arange(len(max_depth_range)), max_depth_range,fontsize='small')
    plt.xticks(np.arange(len(min_samples_leaf_range)),min_samples_leaf_range,fontsize='small',rotation=45)
    plt.title(metric + ' metric ' + '[split criterion : ' + split +']')
    plt.savefig('Graphs/' + metric  + '.png',bbox_inches='tight')
    
def plot_ams_curve(cuts,ams_curve,label):
     
     peak_ams = max(ams_curve)
     best_cut = cuts[ams_curve.index(max(ams_curve))]
     plt.figure()
     plt.grid()
     plt.xlim(min(cuts),max(cuts))
     plt.plot(cuts,ams_curve,'r+-',label='Test AMS')
     plt.axhline(y=peak_ams,label='Peak AMS = '+str(round(peak_ams,2)),color='black',linestyle='--')
     plt.axvline(x=best_cut,linestyle='--',label = 'Best threshold = ' + str(best_cut),color='black')
     plt.xlabel('Threshold (% Rejected)')
     plt.ylabel('AMS Score')
     plt.legend(loc=3)
     plt.title('AMS Curve ' + label) 
     plt.savefig('Graphs/AMS_curve_' + label + '.png',bbox_inches='tight')
     
     
if __name__ == "__main__":
    
    # Loading the dataset
    df = preprocessing.load_data(path='/home/raid/vr308/workspace/Python/higgsDT/Data/')
    df = preprocessing.drop_missing_values(df)
    df_normed = preprocessing.normalize(df)[0]
    df_features = preprocessing.get_features(df)
    
   # Get background/signal samples
    b = df[df.Label == 1]
    s = df[df.Label == 0]
    
    features = df_features.columns
    
    f1 = 'DER_mass_MMC' # Sample feature 1
    f2 = 'PRI_tau_eta' # Sample feature 2
    
    # Feature Heat Map
    title =  'Feature Correlation Heatmap : Primary & Derived Features'
    figsave=True
    
    plot_feature_corr_matrix(df_features,title,figsave)
    
    #Feature-Class Heat Map
    plot_feature_class_corr_matrix(preprocessing.get_features(df),df.Label,features)
    
    # Bi Feature Distribution    
    plot_bi_features(b,s,f1,f2)
    
    f = ['DER_mass_jet_jet','A','DER_deltaeta_jet_jet','DER_prodeta_jet_jet']
    # Histogram of univariate feature 
    for i in f:
        plot_hist(b,s,i,isNorm=False)
    







    