# -*- coding: utf-8 -*-
"""
This class computes and plots the discovery significance (AMS)
It is invoked in the last step of running the main script.

@author: vr308

"""

import numpy as np
import math
import sys
import matplotlib.pylab as plt
import logging
import ConfigParser
settings = ConfigParser.ConfigParser()
settings.read('settings.ini')

logger_discovery = logging.getLogger('discovery_significance.py')
out_handler = logging.StreamHandler(sys.stdout)
logger_discovery.addHandler(out_handler)

class AMS:

    """ This class provides functions to calculate the AMS (sigma) """

    # The mathematical form of this AMS function is derived from
    #http://www.jmlr.org/proceedings/papers/v42/cowa14.pdf.

    @classmethod
    def ams_compute(cls, s, b):

        return  round(math.sqrt(2.0 * ((s + b + 10) * math.log(1.0 + s / (b + 10)) - s)),6)

    @classmethod
    def get_ams_score(cls, W_test, Y_test, Y_test_pred):

        signals = W_test * (Y_test == 1) * (Y_test_pred == 1)
        background = W_test * (Y_test == -1) * (Y_test_pred == 1)
        s = np.sum(signals)
        b = np.sum(background)
        ams = AMS.ams_compute(s, b)
        return ams

    @classmethod
    def plot_ams_curve(cls, thresh, ams_curve, classifier, legend_text, new_fig, save, settings):

        if new_fig:
            plt.figure()
            plt.hlines(y=3.8058,xmin=80,xmax=95,colors='r')
        plt.grid(b=True,which='both',axis='both')
        plt.minorticks_on()
        plt.title('AMS vs. Cut-off')
        plt.plot(thresh, ams_curve, label=legend_text)
        max_thresh = ams_curve.index(max(ams_curve))
        plt.scatter(thresh[max_thresh],max(ams_curve),marker='o',color='r')
        plt.xlabel('Selection Threshold %')
        plt.ylabel('$\sigma$')
        plt.title('AMS ' + '($\sigma$)' + ' for ' + 'Classifier: ' + settings.get('algorithmName', classifier.algorithm))
        plt.legend(prop={'size': 9})
        locs = np.arange(2.5, 4, 0.1)
        labels = labels = map(lambda x: str(x) + '$\sigma$', locs)
        plt.yticks(locs, labels)
        plt.ylim(2.5, 4)
        plt.xlim(80,95)
        plt.tight_layout()
        if save:
            title = 'Graphs/AMS_Curve_' + classifier.algorithm + '.png'
            print 'Saving graph in ' + title
            plt.savefig(title)

    @classmethod
    def ams_curve(cls, W_test, Y_test, test_score, train_score, classifier, new_fig, save, settings):
        ams_curve = []
        thresholds = np.arange(80, 95.01, 0.01)
        algorithm = classifier.algorithm
        logger_discovery.info('Computing AMS ' + u'\u03C3' + ' across thresholds 80 to 90 with a step size of 0.01')
        for i in thresholds:
            cutoff = classifier.get_score_cutoff_for_threshold(train_score, i)
            Y_test_pred = classifier.get_predicted_labels(classifier.X_test, test_score, cutoff)
            ams = AMS.get_ams_score(W_test, Y_test, Y_test_pred)
            ams_curve.append(ams)
            if round(i%1,5) == 0:
                logger_discovery.info('AMS' + ' at ' + str(i) + ' percentile....' + str(ams) + u'\u03C3')
                print('AMS' + ' at ' + str(i) + ' percentile....' + str(ams) + u'\u03C3')
            if (algorithm != 'BDT') and (algorithm != 'BXT') and (algorithm != 'BRF'):
                if algorithm == 'DT':
                    n_trees = 1
                else:
                    n_trees = classifier.trained_classifier.n_estimators
                max_depth = classifier.trained_classifier.max_depth
                max_features = classifier.trained_classifier.max_features
                legend_text = 'n_trees: ' + str(n_trees) + '\n' + 'max_depth: ' + str(max_depth) + '\n' + 'max_features: ' + str(max_features)
            else:
                if algorithm == 'BDT':
                    n_trees = 1
                else:
                    n_trees = classifier.trained_classifier.base_estimator.n_estimators
                n_stages = classifier.trained_classifier.n_estimators
                max_depth = classifier.trained_classifier.base_estimator.max_depth
                max_features = classifier.trained_classifier.base_estimator.max_features
                legend_text = 'boosting_stages: ' + str(n_stages) + '\n' + 'n_trees: ' + str(n_trees) + '\n' + 'max_depth: ' + str(max_depth) + '\n' + 'max_features: ' + str(max_features)
        AMS.plot_ams_curve(thresholds, ams_curve, classifier, legend_text, new_fig, save, settings)
        logger_discovery.removeHandler(out_handler)
        return ams_curve, thresholds