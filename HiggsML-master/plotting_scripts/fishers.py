# -*- coding: utf-8 -*-
"""
@author: vr308
"""

import numpy as np
import sys, os
import ConfigParser
from higgs_data import HiggsData
import matplotlib.pyplot as plt
import scipy.stats as stats
from preprocessing import Preprocessing
import operator
sys.path.append(os.path.join(os.path.realpath('..')))
settings = ConfigParser.ConfigParser()
settings.read('../settings.ini')


def plot_fishers_ratio(b,s):
    fishers = []
    n = len(b.columns)
    for i in np.arange(0,n):
        mu_b = np.mean(b.icol(i))
        mu_s = np.mean(s.icol(i))
        var_b = np.var(b.icol(i))
        var_s = np.var(s.icol(i))
        f = np.square(mu_b - mu_s)/(var_b+var_s)
        fishers.append((b.columns[i],f))
    fishers.sort(key=operator.itemgetter(1))
    plt.figure(figsize=(10,8))
    key_color = map(lambda x: 1 if x.startswith('PRI') else 0, zip(*fishers)[0])
    colors = np.asarray(['c','g'])
    plt.barh(bottom=xrange(34),width=zip(*fishers)[1],color=colors[key_color],alpha=0.5,edgecolor='g')
    plt.yticks(np.arange(34)+0.5,zip(*fishers)[0],fontsize='small')
    plt.title("Ranking by Fisher's Discriminant Score", fontsize='small')
    plt.tight_layout()
    plt.savefig('../Graphs/fishers.png')
    return fishers


def plot_feature_class_corr_matrix(df,labels,cols):
        zero_one_labels = map(lambda x : 0 if x == -1 else 1, labels)
        corr = []
        for i in xrange(0,df.shape[1]):
            corr.append(stats.pointbiserialr(zero_one_labels,df.icol(i)))
        pos = np.arange(1,df.shape[1] +1)
        c, p = zip(*corr)
        tuples = zip(c,list(cols))
        tuples = sorted(tuples,key=lambda x: x[0])
        plt.figure(figsize=(10,10))
        key_color = map(lambda x: 1 if x.startswith('PRI') else 0, zip(*tuples)[1])
        colors = np.asarray(['c','g'])
        plt.barh(bottom=pos,width=zip(*tuples)[0],color=colors[key_color],edgecolor=None,alpha=0.7)
        plt.yticks(np.arange(1,max(pos)+0.5),zip(*tuples)[1],fontsize='small')
        plt.xlabel('Correlation', fontsize='small')
        plt.title('Ranking by point biserial correlation - Features v. Class', fontsize='small')
        plt.savefig('../Graphs/pbc.png')
        plt.tight_layout()

if __name__ == "__main__":

    hd = HiggsData(path=settings.get('paths','path_data'), imputation=True)
    df = Preprocessing.remove_missing_values(hd.processed_input,np.NaN)
    b_processed = Preprocessing.get_features(df[df.Label  == -1])
    s_processed = Preprocessing.get_features(df[df.Label == 1])

    labels = hd.raw_input.ix[df.index]['Label']
    df_features = Preprocessing.get_features(df)
    cols = df_features.columns

    fishers = plot_fishers_ratio(b_processed, s_processed)
    pbc = plot_feature_class_corr_matrix(df_features,labels,cols)

