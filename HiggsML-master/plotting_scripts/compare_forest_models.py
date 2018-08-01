# -*- coding: utf-8 -*-
"""
This script provides functions to generate plots which compare two models:

    1) Random Forests
    2) Extremely Random Trees

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

import timeit
import gzip
import pickle
import numpy as np
import matplotlib.pylab as plt
import sys, os
import pandas as pd
import ConfigParser
import sklearn.ensemble as ensemble
sys.path.append(os.path.join(os.path.realpath('..')))

from discovery_significance import AMS
settings = ConfigParser.ConfigParser()
settings.read('../settings.ini')

def plot_ams_rf_et(thresh, rf_curve, et_curve):

        """ This function generates and saves two overlaid ams curves"""

        plt.close()
        plt.figure()
        plt.hlines(y=3.8058,xmin=80,xmax=95,colors='r')
        plt.grid(b=True,which='both',axis='both')
        plt.minorticks_on()
        plt.title('AMS ' + '($\sigma$)' +  ' vs. Cut-off [Random Forest | Extremely Random Trees]', fontsize='small')
        plt.plot(thresh, rf_curve, label='RF', color='g')
        max_thresh = rf_curve.index(max(rf_curve))
        plt.scatter(thresh[max_thresh],max(rf_curve),marker='o',color='r')
        plt.plot(thresh, et_curve, label='ET', color='b')
        max_thresh = et_curve.index(max(et_curve))
        plt.scatter(thresh[max_thresh],max(et_curve),marker='o',color='r')
        plt.xlabel('Selection Threshold %', fontsize='small')
        plt.ylabel('$\sigma$', fontsize='small')
        plt.legend(prop={'size': 9})
        locs = np.arange(2.5, 4, 0.1)
        labels = labels = map(lambda x: str(x) + '$\sigma$', locs)
        plt.yticks(locs, labels)
        plt.ylim(2.5, 4)
        plt.xlim(80,95)
        plt.tight_layout()
        title = '../Graphs/AMS_Curve_ET_RF' + '.png'
        print 'Saving graph in ' + title
        plt.savefig(title)


def plot_feature_importances(rff, etf):

        """ This function generates and saves two graphs which ranks features by their importance in the
            respective model.
        """
        plt.figure(figsize=(11,8))
        pos = np.arange(1,len(rff)+1)
        tuples = zip(rff, etf, features)
        rf_tuples = sorted(tuples,key=lambda x: x[0])
        et_tuples = sorted(tuples,key=lambda x: x[1])
        plt.hlines(pos,0,zip(*rf_tuples)[0], color='green')  # Stems
        plt.plot(zip(*rf_tuples)[0], pos, 'D', color='r')  # Stem ends
        plt.yticks(pos, zip(*rf_tuples)[2], fontsize='small')
        plt.ylim(0,35)
        plt.title('Random Forest - Feature importances', fontsize='small')
        plt.tight_layout()
        plt.savefig('../Graphs/RF_feature_importances.png')
        plt.figure(figsize=(10,8))
        plt.hlines(pos,0,zip(*et_tuples)[1], color='blue')  # Stems
        plt.plot(zip(*et_tuples)[1], pos, 'D', color='r')  # Stem ends
        plt.yticks(pos, zip(*et_tuples)[2], fontsize='small')
        plt.ylim(0,35)
        plt.title('Extremely Random Trees - Feature importances', fontsize='small')
        plt.tight_layout()
        plt.savefig('../Graphs/ET_feature_importances.png')

def plot_correlation_heat_maps(rf, et):
    """ This graph computes the correlation between the individual tree
        outputs in each of the two models"""
    print 'Computing correlation between tree outputs'
    models = [rf, et]
    correlation_matrices = []
    for k in models:
        n_estimators = xrange(len(k.trained_classifier.estimators_))
        _trees = pd.DataFrame(columns=list(map(lambda x: 'tree_' + str(x), n_estimators)), index=k.X_test.index)
        for model, i in zip(k.trained_classifier.estimators_, _trees.columns):
            _trees[i] = model.predict_proba(k.X_test)[:,1]
        correlation_matrices.append(_trees.corr())
    plt.figure(figsize=(10,8))
    plt.imshow(correlation_matrices[0],interpolation='none', cmap=plt.cm.jet, vmin=0.3, vmax=1)
    plt.colorbar()
    plt.title('Tree Correlation in Random Forests', fontsize='small')
    plt.xticks(n_estimators,np.arange(1,11))
    plt.yticks(n_estimators,np.arange(1,11))
    plt.xlabel('Tree number', fontsize='small')
    plt.ylabel('Tree number', fontsize='small')
    plt.savefig('../Graphs/RF_tree_correlation.png')
    plt.figure(figsize=(10,8))
    plt.imshow(correlation_matrices[1], interpolation='none', cmap=plt.cm.jet, vmin=0.3, vmax=1)
    plt.title('Tree Correlation in Extremely Random Trees', fontsize='small')
    plt.colorbar()
    plt.xticks(n_estimators,np.arange(1,11))
    plt.yticks(n_estimators,np.arange(1,11))
    plt.xlabel('Tree number', fontsize='small')
    plt.ylabel('Tree number', fontsize='small')
    plt.savefig('../Graphs/ET_tree_correlation.png')


def ams_vs_cutoff(X_test, Y_test, W_test, train_score, test_score):
    thresholds = np.arange(80, 95.01, 0.01)
    ams_curve = []
    for i in thresholds:
        cutoff = np.percentile(train_score, i)
        Y_predicted = pd.Series(map(lambda x: 1 if x > cutoff else -1, test_score), index=X_test.index)
        ams = AMS.get_ams_score(W_test, Y_test, Y_predicted)
        ams_curve.append(ams)
    return ams_curve, thresholds


def get_curve_per_model(model):
    _models = [model.trained_classifier]

    runtime = []
    for k in [25, 50, 100, 200]:
        if model.algorithm == 'RF':
            base_model = ensemble.RandomForestClassifier(n_jobs=4, verbose=1)
        elif model.algorithm == 'ET':
            base_model = ensemble.ExtraTreesClassifier(n_jobs=4, verbose=1)
        for i in model.param_ranges.keys():
            setattr(base_model, i, model.trained_classifier.get_params()[i])
        setattr(base_model,'n_estimators', k)
        print base_model
        start = timeit.default_timer()
        base_model.fit(model.X_train, model.Y_train, model.W_train)
        elapsed = timeit.default_timer()
        runtime.append(elapsed-start)
        _models.append(base_model)

    curves = []
    for m in _models:
        print 'Constructing AMS curve for models with n_estimators (this could take several minutes)'
        train_score = m.predict_proba(model.X_train)[:,1]
        test_score = m.predict_proba(model.X_test)[:,1]
        curve, thresh = ams_vs_cutoff(model.X_test, model.Y_test, model.W_test, train_score, test_score)
        curves.append(curve)
    return curves, thresh, runtime

def plot_ams_evolution(curves_rf, curves_et, estimators):

    """ This function plots the evolution of the AMS as the number of trees in the ensemble increases"""
    max_curves = []
    for k in [curves_rf, curves_et]:
        max_ams = []
        for i in xrange(len(k)):
            max_ams.append(max(k[i]))
        max_curves.append(max_ams)
    plt.figure()
    plt.minorticks_on()
    labels = ['RF', 'ET']
    colors = ['g','b']
    for i in [0,1]:
        plt.plot(estimators, max_curves[i], label=labels[i], color=colors[i])
        plt.scatter(estimators, max_curves[i], color='r')
    plt.title('Peak AMS' +  '$(\sigma)$' + ' evolution with number of trees in ensemble', fontsize='small')
    plt.xlabel('No. of trees', fontsize='small')
    plt.ylabel('Peak AMS ' + '$(\sigma)$', fontsize='small')
    locs = np.arange(3.3, 3.65, 0.05)
    labels = map(lambda x: str(x) + '$\sigma$', locs)
    plt.legend(fontsize='small')
    plt.yticks(locs, labels)
    plt.ylim(3.35,3.65)
    plt.xlim(0,220)
    plt.xticks(estimators)
    plt.grid()
    plt.savefig('../Graphs/Peak_AMS_Evolution.png')

def plot_runtime_evolution(runtimes_rf, runtimes_et, estimators):

    """ This function plots the evolution of runtime as the number of trees in the ensemble increases"""
    plt.figure()
    plt.grid(b=True,which='both',axis='both')
    plt.minorticks_on()
    plt.plot(estimators, runtimes_rf, label='RF', color='g')
    plt.plot(estimators, runtimes_et, label='ET', color='b')
    plt.plot(estimators, runtimes_rf, 'D', color='r')
    plt.plot(estimators, runtimes_et, 'D', color='r')
    plt.legend(fontsize='small')
    plt.title('Runtime vs. number of trees [Random Forest | Extremely Random Trees]', fontsize='small')
    plt.yticks(fontsize='small')
    plt.ylabel('Seconds', fontsize='small')
    plt.xlabel('Number of trees', fontsize='small')
    plt.xticks(estimators, fontsize='small')
    plt.xlim(10,220)
    plt.ylim(0, 450)
    plt.savefig('../Graphs/Runtime_evolution_RF_ET.png')


def plot_group_ams(curves, thresholds, estimators, algorithm):

    plt.figure()
    plt.hlines(y=3.8058,xmin=80,xmax=95,colors='r')
    plt.grid(b=True,which='both',axis='both')
    plt.minorticks_on()
    plt.title('AMS ' + '($\sigma$)' +  ' vs. Cut-off [Algorithm: ' + algorithm + ']', fontsize='small')
    for i in xrange(len(estimators)):
        plt.plot(thresholds, curves[i], label='n_trees ' + str(estimators[i]))
        max_thresh = curves[i].index(max(curves[i]))
        plt.scatter(thresholds[max_thresh],max(curves[i]),marker='o',color='magenta')
    plt.xlabel('Selection Threshold %', fontsize='small')
    plt.ylabel('$\sigma$', fontsize='small')
    plt.legend(prop={'size': 9})
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


if __name__ == "__main__":

    # Run this block first to load the pickled models,
    # The results are generated on the basis of these models.

    target = gzip.open('../Pickled/clf_rf_opt.pklz','rb')
    rf = pickle.load(target)
    target.close()

    rf_train_score = rf.get_decision_scores(rf.X_train)
    rf_test_score = rf.get_decision_scores(rf.X_test)

    target = gzip.open('../Pickled/clf_et_opt.pklz','rb')
    et = pickle.load(target)
    target.close()

    et_train_score = et.get_decision_scores(et.X_train)
    et_test_score = et.get_decision_scores(et.X_test)

    ###############################################

    # Generates a plot which shows the AMS curves

    print 'Generating scores from models, (this could take few minutes)'
    rf_curve, thresh = AMS.ams_curve(rf.W_test, rf.Y_test, rf_test_score, rf_train_score, rf, new_fig=True, save=False, settings=settings)
    et_curve, thresh = AMS.ams_curve(et.W_test, et.Y_test, et_test_score, et_train_score, et, new_fig=True, save=False, settings=settings)
    plot_ams_rf_et(thresh, rf_curve,et_curve)

    ###############################################

    print 'Plotting Correlation in tree outputs'

    plot_correlation_heat_maps(rf, et)

    ################################################

    print 'Plotting feature importances'

    features = rf.X_train.columns
    rf_feature_importances = rf.trained_classifier.feature_importances_
    et_feature_importances = et.trained_classifier.feature_importances_
    plot_feature_importances(rf_feature_importances, et_feature_importances)

    ################################################

    print 'Plotting Peak AMS evolution with n_estimators in RF and ET (this could take several minutes)'

    curves_rf, thresh, runtimes_rf = get_curve_per_model(rf)
    curves_et, thresh, runtimes_et = get_curve_per_model(et)

    plot_group_ams(curves_rf, thresh, estimators=[10, 25, 50, 100, 200], algorithm = 'RF')
    plot_group_ams(curves_et, thresh, estimators=[10, 25, 50, 100, 200], algorithm = 'ET')
    plot_runtime_evolution(runtimes_rf, runtimes_et, estimators=[25, 50 ,100, 200])
    plot_ams_evolution(curves_rf, curves_et, estimators=[10, 25, 50, 100, 200])


    #################################################

