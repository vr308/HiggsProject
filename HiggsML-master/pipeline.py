# -*- coding: utf-8 -*-
"""
This scripts provides handler functions to handle requests
by the __main__.py script to run a classifer.

@author: vr308
"""

from higgs_data import HiggsData
from classifier_engine import ClassifierEngine
from performance import PerformanceReports
import timeit
import logging
import sys
import numpy as np

logger_pipeline = logging.getLogger('pipeline')
out_handler = logging.StreamHandler(sys.stdout)
logger_pipeline.addHandler(out_handler)


def timer(start_time):

    return np.round(timeit.default_timer() - start_time, 2)


def load(path, imputation):

    logging_phase = 'Load Data'
    logger_pipeline.info('--------Starting ' + logging_phase + '---------')

    start_timer_load = timeit.default_timer()
    hd = HiggsData(path, imputation)
    elapsed = timer(start_timer_load)

    logger_pipeline.info('--------Finished ' + logging_phase + '---------')
    logger_pipeline.info('*****Load Data took ' + str(elapsed) + ' wall-clock seconds*****')
    return hd


def hyperparameter_tuning(classifier, settings, weighted):

    logging_phase = 'Hyperparameter Tuning'
    logger_pipeline.info('--------Starting ' + logging_phase + '---------')

    start_timer_tuning = timeit.default_timer()
    logger_pipeline.debug('Optimizing balanced classification error for best parameters over a grid')
    cv_scores, classifier, grid = classifier.grid_search(weighted, settings)
    logger_pipeline.debug(classifier.best_params)
    elapsed = timer(start_timer_tuning)

    logger_pipeline.info('---------Finished Tuning----------')
    logger_pipeline.info('*****Tuning took ' + str(elapsed) + ' wall-clock seconds*****')
    return cv_scores, classifier, grid

def read_classifier_parameters(classifier, settings):

    """ Read the parameters of the algorithm specified when creating classifier
        from the settings.ini file.
    """
    algorithm = classifier.algorithm
    parameters = classifier.param_ranges.keys()
    args = {}
    for i in parameters:
        if i == 'max_features':
            # max_features could either be a string specifying a rule like 'sqrt' or an int
            try:
                param_value = settings.getint(algorithm, i)
            except ValueError:
                param_value = settings.get(algorithm, i)
        elif i == 'learning_rate':
            param_value = settings.getfloat(algorithm, i)
        elif i == 'criterion':
            param_value = settings.get(algorithm, i)
        elif i == 'oob_score' or i == 'warm_start' or i == 'bootstrap':
            param_value = settings.getboolean(algorithm, i)
        elif i == 'base_estimator':
            # this indicates that the specified algorithm is an ensemble (either bagged or boosted) and we need to read params of the base learners
            if algorithm == 'BDT':
                base_learner = 'DT'
            elif algorithm == 'BRF':
                base_learner = 'RF'
            elif algorithm == 'BXT':
                base_learner = 'ET'
            base_learner_model = ClassifierEngine._classification_algortihms.get(base_learner)
            base_learner_model.criterion = settings.get(base_learner,'criterion')
            base_learner_model.max_features = settings.getint(base_learner,'max_features')
            base_learner_model.max_depth = settings.getint(base_learner,'max_depth')
            base_learner_model.min_samples_split = settings.getint(base_learner,'min_samples_split')
            base_learner_model.min_samples_leaf = settings.getint(base_learner,'min_samples_leaf')
            if base_learner == 'RF' or base_learner == 'ET':
                base_learner_model.n_estimators = settings.getint(base_learner,'n_estimators')
            param_value = base_learner_model
        else:
            param_value = settings.getint(algorithm, i)
        args.update({i: param_value})
    return args


def train_classifier_engine(classifier, settings, weighted):

    logging_phase = 'Training'
    logger_pipeline.info('--------Starting ' + logging_phase + '---------')

    start_timer_training = timeit.default_timer()
    logger_pipeline.debug('Extracting parameters from settings file')
    args = read_classifier_parameters(classifier, settings)
    logger_pipeline.debug(args)
    classifier.train_classifier(weighted, args)
    elapsed = timer(start_timer_training)

    logger_pipeline.info('---------Finished Training----------')
    logger_pipeline.info('*****Training took ' + str(elapsed) + ' wall-clock seconds*****')
    return classifier

def get_predictions(classifier, X_train, X_test, selection_threshold):

    logger_pipeline.info('---------Starting Testing---------')
    start_timer_testing = timeit.default_timer()
    test_score = classifier.get_decision_scores(X_test)
    train_score = classifier.get_decision_scores(X_train)
    cutoff = classifier.get_score_cutoff_for_threshold(train_score, selection_threshold)
    Y_test_predicted = classifier.get_predicted_labels(X_test, test_score, cutoff)
    elapsed = timer(start_timer_testing)
    logger_pipeline.info('---------Finished Testing---------')
    logger_pipeline.info('*****Testing took ' + str(elapsed) + ' wall-clock seconds*****')
    return train_score, test_score, Y_test_predicted

def derive_selection_region(classifier, X_test, Y_test_predicted):

    selection_region = classifier.get_selection_events(X_test, Y_test_predicted)
    return selection_region


def performance_metrics(classifier, W_test, Y_test, Y_test_pred, selection_region, selection_threshold, W_test_balanced):

    logger_pipeline.info('---------Starting Performance Reporting---------')
    start_timer_pr = timeit.default_timer()

    selection_weights = selection_region[1]
    selection_labels = selection_region[2]

    pr = PerformanceReports()
    pr.prepare_report( W_test, Y_test, Y_test_pred, selection_threshold, classifier, W_test_balanced, selection_weights, selection_labels)

    elapsed = timer(start_timer_pr)

    pr.make_report(classifier.algorithm)
    logger_pipeline.info('---------Finished Performance Reporting-----------'),
    logger_pipeline.info('*****Reporting took ' + str(elapsed) + ' wall-clock seconds*****')
    logger_pipeline.removeHandler(out_handler)
