# -*- coding: utf-8 -*-
"""
This class computes, prints and saves performance reports based
on the output of the classifier engine.
The output is saved in /Results/ sub-directory.

@author: vr308

"""
import sklearn.metrics as metrics
from collections import OrderedDict
from discovery_significance import AMS
import random
import sys
import numpy as np
import logging
import ConfigParser
settings = ConfigParser.ConfigParser()
settings.read('settings.ini')

logger_performance = logging.getLogger('performance.py')
out_handler = logging.StreamHandler(sys.stdout)
logger_performance.addHandler(out_handler)


class PerformanceReports:

    def __init__(self):

        self.report = OrderedDict()

    @classmethod
    def classification_report(cls, true, predicted):

        print metrics.classification_report(true, predicted)

    @classmethod
    def signal_recall(cls, true, predicted):

        recall = metrics.recall_score(true, predicted, pos_label=+1)
        return np.round(recall, 2)

    @classmethod
    def signal_precision(cls, true, predicted):

        precision = metrics.precision_score(true, predicted, pos_label=+1)
        return np.round(precision, 2)

    @classmethod
    def background_precision(cls, true, predicted):

      precision = metrics.precision_score(true, predicted, pos_label=-1)
      return np.round(precision, 2)

    @classmethod
    def background_recall(cls, true, predicted):

      recall = metrics.recall_score(true,predicted, pos_label=-1)
      return np.round(recall, 2)

    @classmethod
    def roc_auc_score(cls,true,predicted):

      roc_auc_score = metrics.roc_auc_score(true, predicted)
      return np.round(roc_auc_score, 2)

    @classmethod
    def roc_curve(cls, true, predicted):

      fpr,tpr,thresh = metrics.roc_curve(true, predicted)
      return fpr,tpr,thresh

    @classmethod
    def confusion_matrix(cls, true, predicted):

      cf = metrics.confusion_matrix(true, predicted)
      return cf

    @classmethod
    def likelihood_ratio(cls, true, predicted):

        cm = PerformanceReports.confusion_matrix(true, predicted)
        true_positive_rate = float(cm[1][1]) / (cm[1][1] + cm[1][0])
        false_positive_rate = float(cm[0][1]) / (cm[0][1] + cm[0][0])
        return np.round(true_positive_rate / false_positive_rate, 2)

    @classmethod
    def balanced_classification_error(cls, true, predicted, balanced_weights):

        return round(np.sum(balanced_weights * (true != predicted)), 4) * 100

    def prepare_report(self, W_test, Y_test, Y_test_predicted, selection_threshold, classifier, W_test_balanced, selection_weights,selection_labels):

        select_predicted = Y_test_predicted[Y_test_predicted == 1]
        true_positives = len(selection_labels[selection_labels == 1])
        false_positives = len(selection_labels[selection_labels == -1])
        selection_size = len(selection_labels)

        self.report.update({'Classifier Acronym': classifier.algorithm})
        self.report.update({'Parameters ': classifier.best_params})
        self.report.update({'Selection Threshold [percentile]': selection_threshold})
        self.report.update({'Number of Selection Events ': selection_size})
        self.report.update({'True Positives (Selection Region)': true_positives})
        self.report.update({'False Positives (Selection Region)': false_positives})
        self.report.update({'SignalToBackgroundRatio ': np.round(true_positives / false_positives, 4)})
        self.report.update({'Selection Error / False positive rate ': round(false_positives / (selection_size * 1.0), 4)})
        self.report.update({'Signal Recall ': round(true_positives / (selection_size * 1.0), 4)})
        self.report.update({'Balanced Classification Error %': PerformanceReports.balanced_classification_error(Y_test,Y_test_predicted, W_test_balanced)})
        self.report.update({'Signal Precision ': PerformanceReports.signal_precision(selection_labels, select_predicted)})
        self.report.update({'Discovery Significance ': AMS.get_ams_score(W_test, Y_test, Y_test_predicted)})
        #self.report.update({'Signal Recall ' : PerformanceReports.signal_recall(true,predicted)})
        #self.report.update({'Background Precision ' : PerformanceReports.background_precision(true,predicted)})
        #self.report.update({'Background Recall ' :  PerformanceReports.background_recall(true,predicted)})
        #self.report.update({'ROC AUC Score ' : PerformanceReports.roc_auc_score(true,predicted)})
        #self.report.update({'Likelihood Ratio (TPR/FPR) ' : PerformanceReports.likelihood_ratio(true,predicted)})

    def print_report(self):

        width = 40
        for i in self.report:
            key = str(i)
            value = str(self.report[i])
            if key == 'Discovery Significance ':
                 value = value + u'\u03C3'
                 value = value.encode('utf-8')
            string = "{}| {}".format(key.ljust(width),value.ljust(width))
            logger_performance.info(string)
            print string

    def make_report(self,algorithm):

        filename = 'Results/Classifier_Performance_Report_' + algorithm + '_' + str(random.randint(1,100)) + '.txt'
        logger_performance.info('---------Printing Performance Report--------------')
        headline = 'Classifier Performance for  ' + algorithm + ': ' + str(settings.get('algorithmName',algorithm)) + '\n'
        sys.stdout = open(filename,'w')
        logger_performance.info(headline)
        print headline
        self.print_report()
        logger_performance.info('Writing performance report to file : ' + filename + '\n')
        logger_performance.removeHandler(out_handler)
        sys.stdout.close()
        sys.stdout = sys.__stdout__