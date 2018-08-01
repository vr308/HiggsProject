# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:02:25 2016

@author: vr308
"""

"""
This script provides functions to generate plots which compare two models:

    1) Boosted Random Forests
    2) Boosted Extremely Random Forests

The plots are all saved in /Graphs/ subdirectory.

Instead of running the script at once, it is suggested to load the functions before __main__
and then execute the blocks which call the functions.

Under __main__ each block is demarcated by a line of '#'
This scripts uses relative imports to save plots in a folder (/Graphs/)
which lies in an upper level directory.
It assumes when the script is run that the current working directory
is in the directory in which it resides.

They rely on serialized pre-trained models which have been included in the Pickled/ folder

@author: vr308
"""

import gzip
import pickle
import math
import numpy as np
import matplotlib.pylab as plt
import sys, os
import pandas as pd
import sklearn.ensemble as ensemble
import ConfigParser
sys.path.append(os.path.join(os.path.realpath('..')))
settings = ConfigParser.ConfigParser()
settings.read('../settings.ini')
from classifier_engine import ClassifierEngine
from discovery_significance import AMS

def plot_ams_brf_bxt(thresh, brf_curve, bxt_curve):
        """ This function generates and saves two overlaid ams curves"""
        plt.close()
        plt.figure()
        plt.hlines(y=3.8058,xmin=80,xmax=95,colors='r')
        plt.grid(b=True,which='both',axis='both')
        plt.minorticks_on()
        plt.title('AMS ' + '($\sigma$)' +  ' vs. Cut-off [BRF | BXT]', fontsize='small')
        plt.plot(thresh, brf_curve, label='BRF', color='g')
        max_thresh = brf_curve.index(max(brf_curve))
        plt.scatter(thresh[max_thresh],max(brf_curve),marker='o',color='r')
        plt.plot(thresh, bxt_curve, label='BXT', color='b')
        max_thresh = bxt_curve.index(max(bxt_curve))
        plt.scatter(thresh[max_thresh],max(bxt_curve),marker='o',color='r')
        plt.xlabel('Selection Threshold %', fontsize='small')
        plt.ylabel('$\sigma$', fontsize='small')
        plt.legend(fontsize='small')
        locs = np.arange(2.5, 4, 0.1)
        labels = labels = map(lambda x: str(x) + '$\sigma$', locs)
        plt.yticks(locs, labels)
        plt.ylim(2.5, 4)
        plt.xlim(80,95)
        plt.tight_layout()
        title = '../Graphs/AMS_Curve_BXT_BRF' + '.png'
        print 'Saving graph in ' + title
        plt.savefig(title)

def plot_forest_correlation_heat_maps(brf, bxt):

    """ This graph computes the correlation between the individual tree
        outputs in each of the two models"""

    print 'Computing correlation between forest outputs'
    models = [brf, bxt]
    correlation_matrices = []
    for k in models:
        n_estimators = xrange(len(k.trained_classifier.estimators_))
        _forests = pd.DataFrame(columns=list(map(lambda x: 'forest_' + str(x), n_estimators)), index=k.X_test.index)
        for model, i in zip(k.trained_classifier.estimators_, _forests.columns):
            _forests[i] = model.predict_proba(k.X_test)[:,1]
        correlation_matrices.append(_forests.corr())
    plt.figure(figsize=(10,8))
    plt.imshow(correlation_matrices[0],interpolation='none', cmap=plt.cm.jet, vmin=0.0, vmax=1)
    plt.colorbar()
    plt.title('Correlation between random forests trained in stages')
    plt.xticks(n_estimators,np.arange(1,21))
    plt.yticks(n_estimators,np.arange(1,21))
    plt.xlabel('Forest number')
    plt.ylabel('Forest number')
    plt.savefig('../Graphs/BRF_forest_correlation.png')
    plt.figure(figsize=(10,8))
    plt.imshow(correlation_matrices[1], interpolation='none', cmap=plt.cm.jet, vmin=0.0, vmax=1)
    plt.title('Correlation between forests of extremely random trees trained in stages')
    plt.colorbar()
    plt.xticks(n_estimators,np.arange(1,21))
    plt.yticks(n_estimators,np.arange(1,21))
    plt.xlabel('Forest number')
    plt.ylabel('Forest number')
    plt.savefig('../Graphs/BXT_forest_correlation.png')


def plot_tree_correlation_heat_maps(brf, bxt):
    """ This graph computes the correlation between the individual tree
        outputs in each of the two models"""
    print 'Computing correlation between tree outputs in a boosted ensemble (this may take several minutes)'
    models = [brf, bxt]
    correlation_matrices = []
    for k in models:
        n_estimators = xrange(100)
        _trees = pd.DataFrame(columns=list(map(lambda x: 'trees_' + str(x),n_estimators)), index=k.X_test.index[1000:2000])
        model = k.trained_classifier.estimators_[0]
        i = 0
        for tree in model.estimators_:
              _trees['trees_' + str(i)] = tree.predict_proba(k.X_test[1000:2000])[:,1]
              i = i + 1
        correlation_matrices.append(_trees.corr())
    plt.figure(figsize=(10,8))
    plt.imshow(correlation_matrices[0],interpolation='none', cmap=plt.cm.jet, vmin=0.0, vmax=1)
    plt.colorbar(shrink=0.7)
    plt.title('Correlation between trees in boosted random forests',fontsize='medium')
    plt.xlabel('Tree number', fontsize='small')
    plt.ylabel('Tree number', fontsize='small')
    plt.savefig('../Graphs/BRF_tree_correlation.png')
    plt.figure(figsize=(10,8))
    plt.imshow(correlation_matrices[1], interpolation='none', cmap=plt.cm.jet, vmin=0.0, vmax=1)
    plt.title('Correlation between trees boosted extremely random trees', fontsize='medium')
    plt.colorbar(shrink=0.7)
    plt.xlabel('Tree number', fontsize='small')
    plt.ylabel('Tree number', fontsize='small')
    plt.savefig('../Graphs/BXT_tree_correlation.png')

def ams_vs_cutoff(X_test, Y_test, W_test, train_score, test_score):
    thresholds = np.arange(80, 95.01, 0.01)
    ams_curve = []
    for i in thresholds:
        cutoff = np.percentile(train_score, i)
        Y_predicted = pd.Series(map(lambda x: 1 if x > cutoff else -1, test_score), index=X_test.index)
        ams = AMS.get_ams_score(W_test, Y_test, Y_predicted)
        ams_curve.append(ams)
    return ams_curve, thresholds


def get_curves_per_stage(model, test_score_staged, train_score_staged, stages):
      curves = []
      for m in stages:
        print 'Constructing AMS curve for models with n_stages (this could take several minutes)'
        curve, thresh = ams_vs_cutoff(model.X_test, model.Y_test, model.W_test, train_score_staged[m], test_score_staged[m])
        curves.append(curve)
      return curves

def plot_ams_evolution(curves_brf, curves_bxtt, stages):
    """ This function plots the evolution of the AMS as the number of stages increases"""
    max_curves = []
    for k in [curves_brf, curves_bxt]:
        max_ams = []
        for i in xrange(len(k)):
            max_ams.append(max(k[i]))
        max_curves.append(max_ams)
    plt.figure()
    plt.minorticks_on()
    labels = ['BRF', 'BXT']
    colors = ['g','b']
    for i in [0,1]:
        plt.plot(stages, max_curves[i], label=labels[i], color=colors[i])
        plt.scatter(stages, max_curves[i], color='r')
    plt.title('Peak AMS' +  '$(\sigma)$' + ' evolution with number of boosting stages', fontsize='small')
    plt.xlabel('No. of stages of boosting', fontsize='small')
    plt.ylabel('Peak AMS ' + '$(\sigma)$', fontsize='small')
    locs = np.arange(3.3, 4, 0.05)
    labels = map(lambda x: str(x) + '$\sigma$', locs)
    plt.legend(fontsize='small')
    plt.yticks(locs, labels)
    plt.ylim(3.30,4)
    plt.xlim(0,21)
    plt.xticks(stages)
    plt.grid()
    plt.savefig('../Graphs/Peak_AMS_Evolution_BRF_BXT.png')

def plot_diff_curve(bxt_curve, brf_curve, thresh):

    diff = np.subtract(bxt_curve, brf_curve)
    plt.figure()
    plt.plot(thresh, diff)
    plt.grid()
    locs = np.arange(0.0, 0.2,0.05)
    labels = labels = map(lambda x: str(x) + '$\sigma$', locs)
    plt.yticks(locs, labels)
    plt.ylim(-0.2,0.2)
    plt.title('Edge of Extremely Randomized trees to Random Forests')


def plot_group_ams(curves, thresholds, stages, algorithm):
    plt.figure()
    plt.hlines(y=3.8058,xmin=80,xmax=95,colors='r')
    plt.grid(b=True,which='both',axis='both')
    plt.minorticks_on()
    plt.title('AMS ' + '($\sigma$)' +  ' vs. Cut-off [Algorithm: ' + algorithm + ']', fontsize='small')
    for i in xrange(len(stages)):
        plt.plot(thresholds, curves[i], label='n_stages ' + str(stages[i]+1))
        max_thresh = curves[i].index(max(curves[i]))
        plt.scatter(thresholds[max_thresh],max(curves[i]),marker='o',color='magenta')
    plt.xlabel('Selection Threshold %', fontsize='small')
    plt.ylabel('$\sigma$', fontsize='small')
    plt.legend(fontsize='small')
    locs = np.arange(2.5, 4, 0.1)
    labels = map(lambda x: str(x) + '$\sigma$', locs)
    plt.legend(fontsize='small')
    plt.yticks(locs, labels)
    plt.ylim(2.5, 4)
    plt.xlim(80,95)
    plt.tight_layout()
    title = '../Graphs/AMS_Curve_' + algorithm + '_Cluster' + '.png'
    print 'Saving graph in ' + title
    plt.savefig(title)

def plot_selection_metrics(brf, bxt, brf_train_score_staged, brf_test_score_staged, bxt_train_score_staged, bxt_test_score_staged):
    ssize_brf, false_positives_brf, fp_weight_brf, tp_weight_brf, ams_stages_brf=get_selection_metrics(brf_train_score_staged, brf_test_score_staged, brf)
    ssize_bxt, false_positives_bxt, fp_weight_bxt, tp_weight_bxt, ams_stages_bxt=get_selection_metrics(bxt_train_score_staged, bxt_test_score_staged, bxt)
    plt.figure()
    plt.plot(xrange(20), false_positives_brf, color='g', label='BRF')
    plt.plot(xrange(20), false_positives_bxt, label = 'BXT')
    plt.title('Count of false positives vs. Boosting stages', fontsize='small')
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    plt.xlabel('No. of stages of boosting', fontsize='small')
    plt.ylabel('False positives', fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize='small')
    plt.savefig('../Graphs/fp_counts.png')

    plt.figure()
    plt.plot(xrange(20),fp_weight_brf, color='g',label='BRF')
    plt.plot(xrange(20),fp_weight_bxt, label='BXT')
    plt.title('Sum across weights of false positives', fontsize='small')
    plt.grid()
    plt.xlabel('No. of stages of boosting', fontsize='small')
    plt.ylabel('Weight of false positives', fontsize='small')
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig('../Graphs/fp_weights.png')

    plt.figure()
    plt.plot(xrange(20), ams_stages_brf, color='g', label='BRF')
    plt.plot(xrange(20), ams_stages_bxt, label = 'BXT')
    plt.title('AMS at 85th percentile threhsold vs. Boosting stages', fontsize='small')
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    plt.xlabel('No. of stages of boosting', fontsize='small')
    plt.ylabel('AMS ' + '$(\sigma)$', fontsize='small')
    plt.grid()
    plt.legend(fontsize='small')
    plt.ylim(3.5, 3.85)
    plt.tight_layout()
    plt.savefig('../Graphs/ams_evolve.png')


def get_selection_metrics(train_score_staged, test_score_staged, model):
    selection_size = []
    false_positives = []
    fp_weight = []
    tp_weight = []
    ams_stages = []
    for i in xrange(20):
        cutoff = np.percentile(train_score_staged[i], 85)
        events = pd.Series(test_score_staged[i], index=model.X_test.index)
        selection_events = events[events > cutoff]
        ids = selection_events.index
        selection_size.append(len(selection_events))
        false_positives.append(len(selection_events[model.Y_test == -1]))
        selection_weights = model.W_test.ix[ids]
        b = np.sum(selection_weights[model.Y_test == -1])
        s = np.sum(selection_weights[model.Y_test == 1])
        ams = round(math.sqrt(2.0 * ((s + b + 10) * math.log(1.0 + s / (b + 10)) - s)),6)
        fp_weight.append(b)
        tp_weight.append(s)
        ams_stages.append(ams)
    return selection_size, false_positives, fp_weight, tp_weight, ams_stages


def plot_staged_decision_scores(test_score_staged, title_text, save_text, weighted, W_test):
    plt.figure()
    for i in [0,4,9,14,19]:
        if weighted == False:
            tag = ''
            plt.hist(test_score_staged[i], bins=100, histtype='stepfilled', alpha=0.5, label = 'stage:' + str(i+1))
            plt.ylabel('Count of events', fontsize='small')
        else:
            plt.hist(test_score_staged[i], bins=100, histtype='stepfilled', weights=W_test,normed=True, alpha=0.5, label = 'stage:' + str(i+1))
            tag = 'weighted'
            plt.ylabel('Sum of importance weights', fontsize='small')
    plt.title('Boosting ' + title_text +  ' - staged discriminant score', fontsize='small')
    plt.legend(title='Boosting stages', fontsize='small', loc=2)
    plt.xlabel('Discriminant score ' + '$M_{h}(\mathbf{x})$', fontsize='small')
    plt.tight_layout()
    plt.savefig('../Graphs/' + save_text + '_' + tag + '_staged.png')

def bifurcated_staged_decision_scores(test_score_staged, title_text, save_text, Y_test, W_test):
     plt.figure(figsize=(12,8))
     for i,j in zip([0,1,4, 9, 15, 19], xrange(6)):
        plt.subplot(str(23) + str(j+1))
        plt.hist(test_score_staged[i][Y_test == -1], bins=100, weights=W_test[Y_test == -1], normed=True,histtype='stepfilled', alpha=0.5)
        plt.hist(test_score_staged[i][Y_test == 1], bins=100, weights=W_test[Y_test == 1], normed=True, color='r', histtype='stepfilled',alpha=0.5)
        cutoff = np.percentile(test_score_staged[i], 85)
        plt.vlines(x=cutoff, ymin=0, ymax=3.2,color='r')
        plt.xticks(fontsize='small')
        plt.yticks(fontsize='small')
        plt.ylim(0,3.2)
        if i == 19:
            plt.title('stage:' + str(20), fontsize='small')
        else:
            plt.title('stage:' + str(i+1), fontsize='small')
     plt.suptitle('Boosting ' + title_text +  ' - staged discriminant score', fontsize='small')
     plt.savefig('../Graphs/' + save_text + '_staged_bi.png')

if __name__ == "__main__":

    # Run this block first to load the pickled models,
    # The results are generated on the basis of these models.

    target = gzip.open('../Pickled/clf_bxt_opt.pklz','rb')
    bxt = pickle.load(target)
    target.close()

    X_test = bxt.X_test
    X_train = bxt.X_train
    Y_test = bxt.Y_test
    Y_train = bxt.Y_train
    W_test = bxt.W_test
    W_train = bxt.W_train
    W_test_balanced = bxt.W_test_balanced
    W_train_balanced = bxt.W_train_balanced

    target = gzip.open('../Pickled/clf_brf_opt.pklz','rb')
    brf_classifier = pickle.load(target)
    target.close()

    brf = ClassifierEngine(X_train, X_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, 'BRF')
    brf.trained_classifier = brf_classifier

    brf_train_score = brf.get_decision_scores(X_train)
    brf_test_score = brf.get_decision_scores(X_test)
    bxt_train_score = bxt.get_decision_scores(X_train)
    bxt_test_score = bxt.get_decision_scores(X_test)

    n_stages_bxt =  xrange(len(bxt.trained_classifier.estimators_))
    staged_bxt_test = bxt.trained_classifier.staged_decision_function(X_test)
    staged_bxt_train = bxt.trained_classifier.staged_decision_function(X_train)

    n_stages_brf =  xrange(len(brf.trained_classifier.estimators_))
    staged_brf_test = brf.trained_classifier.staged_decision_function(X_test)
    staged_brf_train = brf.trained_classifier.staged_decision_function(X_train)

    bxt_test_score_staged = []
    brf_test_score_staged = []
    brf_train_score_staged = []
    bxt_train_score_staged = []

    for i in n_stages_bxt:
        brf_test_score_staged.append(staged_brf_test.next())
        brf_train_score_staged.append(staged_brf_train.next())
        bxt_test_score_staged.append(staged_bxt_test.next())
        bxt_train_score_staged.append(staged_bxt_train.next())

    ###############################################

    print 'Plotting decision scores / staged decision scores '

    plot_staged_decision_scores(bxt_test_score_staged, 'Extremely Random trees','bxt',True, W_test)
    plot_staged_decision_scores(brf_test_score_staged, 'Random Forests', 'brf',True, W_test)
    plot_staged_decision_scores(bxt_test_score_staged, 'Extremely Random trees','bxt',False, W_test)
    plot_staged_decision_scores(brf_test_score_staged, 'Random Forests', 'brf',False, W_test)
    bifurcated_staged_decision_scores(bxt_test_score_staged, 'Extremely Random trees', 'bxt' , Y_test, W_test)
    bifurcated_staged_decision_scores(brf_test_score_staged, 'Random Forest', 'brf', Y_test, W_test)

    ###############################################

    print 'Plotting Correlation in forest and tree outputs'

    plot_forest_correlation_heat_maps(brf, bxt)
    plot_tree_correlation_heat_maps(brf, bxt)

    print 'Plotting staged selection region metrics '

    plot_selection_metrics(brf, bxt, brf_train_score_staged, brf_test_score_staged, bxt_train_score_staged, bxt_test_score_staged)

    # Generates a plot which shows the AMS curves

    print 'Generating scores from models, (this could take few minutes)'

    brf_curve, thresh = AMS.ams_curve(brf.W_test, brf.Y_test, brf_test_score, brf_train_score, brf, new_fig=True, save=False, settings=settings)
    bxt_curve, thresh = AMS.ams_curve(bxt.W_test, bxt.Y_test, bxt_test_score, bxt_train_score, bxt, new_fig=False, save=False, settings=settings)
    plot_ams_brf_bxt(thresh, brf_curve, bxt_curve)

    ###############################################

    print 'Plotting Peak AMS evolution with n_estimators in RF and ET (this could take several minutes)'

    curves_brf = get_curves_per_stage(brf, brf_test_score_staged, brf_train_score_staged, stages=[0,4,9,14,19])
    curves_bxt = get_curves_per_stage(bxt, bxt_test_score_staged, bxt_train_score_staged, stages=[0,4,9,14,19])

    plot_group_ams(curves_brf, thresh, stages=[0,4,9,14,19], algorithm = 'BRF')
    plot_group_ams(curves_bxt, thresh, stages=[0,4,9,14,19], algorithm = 'BXT')
    plot_ams_evolution(curves_brf, curves_bxt, stages=[1,5,10,15,20])

