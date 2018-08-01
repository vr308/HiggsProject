# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 18:33:27 2016

@author: vr308
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from higgs_data import HiggsData
from preprocessing import Preprocessing
import matplotlib.pyplot as plt
import sklearn.metrics as metrics



def naive_mass_classifier(feature):
    return np.log(1/(np.abs(feature - 125)+0.0000001))

def selection_events(score, thresh,test):
    cutoff = np.percentile(score,thresh)
    return test[score > cutoff]

def get_predicted_labels(score, thresh):
    cutoff = np.percentile(score,thresh)
    y_pred = map(lambda x: -1 if x < cutoff else +1, score)
    return np.asarray(y_pred)

def performance(score,thresh,test):
    y_true = test['Label']
    y_pred = get_predicted_labels(score, thresh)
    roc_auc = metrics.roc_auc_score(y_true,y_pred)
    acc = metrics.accuracy_score(y_true,y_pred)
    test_select = selection_events(score,thresh,test)
    true_positives = test_select[test_select.Label == 1]
    false_positives = test_select[test_select.Label == -1]
    s = sum(true_positives['KaggleWeight'])
    b = sum(false_positives['KaggleWeight'])
    return acc,roc_auc,s/np.sqrt(b)

def plot_selection_region(score,threshold):

    plt.figure()
    plt.grid(b=True,which='both',axis='both')
    n,bins,patches=plt.hist(score, bins=100,alpha=0.6,histtype='stepfilled')
    cutoff = np.percentile(score,threshold)
    x_select = bins[bins > cutoff][:-1]
    plt.hist(score[score>x_select[0]],bins=len(x_select),color='r',histtype='stepfilled')
    plt.ylabel('No. of Events')
    plt.vlines(x=x_select[0], ymin=0,ymax=max(n),color='r',label='cut-off ' + r'$\theta_{85}$')
    plt.xlabel(r'$f(\mathbf{x}) = ln\left(\frac{1}{|\mathbf{x} - 125|}\right)$',fontsize=15)
    plt.annotate(s='Predicted ' + '\n' + 'signals',xy=(0,30000))
    plt.annotate(s='(Selection Region)',xy=(0,25000))
    plt.annotate(s=r'$\mathcal{H} = \{\mathbf{x} : f(\mathbf{x}) > \theta_{85}\}$',xy=(5,30000),fontsize='large')
    plt.annotate(s='Predicted ' + '\n' + 'background',xy=(-9,30000))
    plt.title("Distribution of discriminant function " + r'$f(\mathbf{x})$')
    plt.legend()
    plt.tight_layout()

def true_score_separation(score,threshold):

    cutoff = np.percentile(score,threshold)
    b_score = score[test['Label'] == -1]
    s_score = score[test['Label'] == 1]
    plt.figure()
    plt.grid()
    nb,bbins,patches=plt.hist(b_score,bins=100,color='b',alpha=0.6,histtype='stepfilled',label='background')
    ns,sbins,patches=plt.hist(s_score,bins=100,color='r',alpha=0.4,histtype='stepfilled',label='signals')
    plt.vlines(x=cutoff,ymin=0,ymax=25000,label='cut-off ' + r'$\theta_{85}$',color='r')
    plt.annotate(s='(Selection Region)',xy=(0,15000))
    plt.title("Distribuiton of discriminant function " + r'$f(\mathbf{x})$' +'  bifurcated by signal and background events',fontsize='small')
    plt.legend()
    plt.xlabel(r'$f(\mathbf{x})$')
    plt.ylabel('No. of events')
    plt.tight_layout()

def feature_distribution(feature,label):

    plt.figure()
    plt.grid()
    plt.hist(np.log(feature)[label == -1],histtype='stepfilled',color='b',alpha=0.7,bins=100,label='background')
    plt.hist(np.log(feature)[label == +1],histtype='stepfilled',color='r',alpha=0.4,bins=100,label='signal')
    plt.title('Distribution of the natural logarithm of the DER_mass_MMC feature',fontsize='medium')
    plt.ylabel('No. of events')
    plt.xlabel(r'$\ln(\mathbf{x})$')
    plt.legend()
    plt.tight_layout()

def plot_cutoffs(score,test):

    cutoffs = []
    thresh = [75,85,95]
    for i in thresh:
        cutoffs.append(np.percentile(score,i))
    b_score = score[test['Label'] == -1]
    s_score = score[test['Label'] == 1]
    plt.figure()
    plt.grid()
    nb,bbins,patches=plt.hist(b_score,bins=100,color='b',alpha=0.6,histtype='stepfilled')
    ns,sbins,patches=plt.hist(s_score,bins=100,color='r',alpha=0.4,histtype='stepfilled')
    colors = ['b','g','r']
    label = [r'$\theta_{75}$',r'$\theta_{85}$',r'$\theta_{95}$']
    for i in [0,1,2]:
        plt.vlines(x=cutoffs[i],ymin=0,ymax=25000,color=colors[i],label=label[i])
    plt.annotate(s='(Selection Region)',xy=(0,15000))
    plt.annotate(s=r'$\rightarrow$',xy=(0,17000))
    plt.title("Distribuiton of discriminant function " + r'$f(\mathbf{x})$',fontsize='medium')
    plt.legend(title='cut-offs')
    plt.xlabel(r'$f(\mathbf{x})$')
    plt.ylabel('No. of events')
    plt.tight_layout()

def ams_curve_vs_cutoffs(score, test):

    ams=[]
    thresh = [84.5,85,85.5]
    for i in thresh:
        ams.append(performance(score,i,test)[2])
    return ams

if __name__ == "__main__":

    hd = HiggsData()

    test = hd.raw_input[hd.raw_input.KaggleSet == 'v']
    test = test[['DER_mass_MMC','KaggleWeight','Label']]

    missing_ids = test[test['DER_mass_MMC'] == -999.0].index
    test['DER_mass_MMC'][test.index.isin(missing_ids)] = np.NaN
    test = test.dropna(how='any',axis=0)

    score = naive_mass_classifier(test['DER_mass_MMC'])


#acc_curve = []
#ams_curve = []
#fp_count = []
#roc_auc = []
#xticks = np.arange(10,95,1)
#for i in xticks:
#    acc, roc,ams, fp = performance(score,i,test)
#    acc_curve.append(acc)
#    ams_curve.append(ams)
#    roc_auc.append(roc)
#    fp_count.append(fp)
#
#plt.figure()
#plt.plot(xticks,ams_curve)
#
#plt.figure()
#plt.plot(xticks,acc_curve)
#plt.plot(xticks,roc_auc)
#plt.plot(xticks,fp_count)
#
#plt.plot(ams_curve,acc_curve)
#

