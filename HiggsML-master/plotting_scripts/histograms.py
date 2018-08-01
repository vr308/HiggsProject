# -*- coding: utf-8 -*-
"""
This script was used to generate histograms on the basis of raw data
in the higgs dataset.

@author: vr308
"""

import matplotlib.pyplot as plt
import numpy as np
import ConfigParser
import sys, os
sys.path.append(os.path.join(os.path.realpath('..')))
settings = ConfigParser.ConfigParser()
settings.read('../settings.ini')

from higgs_data import HiggsData
from preprocessing import Preprocessing

def histogram_of_weights(b,s,feature):
    b_clean = b[(b[feature] != -999.0)]
    s_clean = s[(s[feature] != -999.0)]
    b_central = reject_outliers(b_clean,feature)
    s_central = reject_outliers(s_clean,feature)
    cumb_weights, bins,patches = plt.hist(b_central[feature].values,bins=100,color='b',weights=b_central['Weight'],normed=True,alpha=0.3,histtype='stepfilled')
    cums_weights, bins, patches = plt.hist(s_central[feature].values,bins=100,color='r',weights=s_central['Weight'],normed=True,alpha=0.6,histtype='stepfilled')
    plt.yticks(fontsize='small')
    plt.xticks(fontsize='small')
    plt.title(feature,fontsize='medium')
    plt.ylim(0,max(max(cumb_weights),max(cums_weights)))
    plt.ylabel('Binned Weights',fontsize='small')

def reject_outliers(df,feature):
    mean = np.mean(df[feature])
    sd = np.std(df[feature])
    return df[(df[feature] < (mean+3*sd)) & (df[feature] > (mean-3*sd))]

def multiplot_histogram(b,s,cols,title_tag,savefig):
    plt.figure(figsize=(10,13))
    multiplot_range = xrange(9)
    for i in multiplot_range:
        plt.subplot(str(33)+str(i+1))
        histogram_of_weights(b,s,cols[i])
    plt.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(left=0.11, right=0.95, top=0.91, bottom=0.06)
    plt.suptitle(title_tag + ' feature distributions of Signal and Background events')
    filename = '../Graphs/' + savefig + '1.png'
    print 'Saving plot in ' + filename
    plt.savefig(filename)
    plt.figure(figsize=(10,13))
    col_range = np.arange(9,len(cols))
    fig_range = xrange(len(col_range))
    for i,j in zip(fig_range,col_range):
        plt.subplot(str(33)+str(i+1))
        histogram_of_weights(b,s,cols[j])
    plt.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(left=0.11, right=0.95, top=0.91, bottom=0.06)
    plt.suptitle(title_tag + ' feature distributions of Signal and Background events')
    filename = '../Graphs/' + savefig + '2.png'
    print 'Saving plot in ' + filename
    plt.savefig(filename)

def multiplot_histogram_phi_features(b,s,phi,savefig):
    plt.figure(figsize=(12,5))
    multiplot_range = xrange(4)
    for i in multiplot_range:
        plt.subplot(str(14)+str(i+1))
        histogram_of_weights(b,s,phi[i])
    plt.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(left=0.11, right=0.95, top=0.91, bottom=0.06)
    filename = '../Graphs/' + savefig + '.png'
    print 'Saving plot in ' + filename
    plt.savefig(filename)

if __name__ == "__main__":


    print 'Loading data'
    hd = HiggsData(path = settings.get('paths','path_data'), imputation=False)
    b_raw = hd.raw_input[hd.raw_input.Label == -1]
    s_raw = hd.raw_input[hd.raw_input.Label == 1]
    b_processed = hd.processed_input[hd.processed_input.Label == -1]
    s_processed = hd.processed_input[hd.processed_input.Label == 1]
    der = Preprocessing.der_features(hd.raw_input)
    pri = Preprocessing.pri_features(hd.raw_input)
    dcols = der.columns
    pcols = pri.columns

    multiplot_histogram(b_raw,s_raw,dcols,'Raw',savefig='der_features')
    multiplot_histogram(b_processed,s_processed,dcols,'Log transformed',savefig='der_features_log_normal')

    phi = [cols for cols in pcols if cols.startswith('PRI_radian')]
    multiplot_histogram_phi_features(b_raw,s_raw,phi,'radian_dist')