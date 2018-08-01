# -*- coding: utf-8 -*-
"""
This is the main function of the software.

It gets its parameters by reading the settings.ini file.
It is not necessary to touch any part of the source code
All controls are through the settings.ini file

The software writes out logs to the Logs/ folder
Graphics to the Graphs/ folder and
Results to the Results/ folder

@author: vr308
"""

import ConfigParser
import warnings
import time
import sys
import gc
import timeit
import logging

from classifier_engine import ClassifierEngine
from discovery_significance import AMS
import pipeline


if __name__ == "__main__":

    run_identifier = int(time.time())
    settings = ConfigParser.ConfigParser()
    settings.read('settings.ini')

    logFile = 'Logs/higgs_classification_pipeline_id_' + str(run_identifier)
    logging.basicConfig(filename=logFile, level=logging.DEBUG, format='%(levelname)s: %(name)s: %(message)s\n')
    logger = logging.getLogger('__main__.py')
    out_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(out_handler)

    logger.info('Reading settings.ini file for configuration')
    logger.info('Writing log stream to file ' + logFile)

    path = settings.get('paths','path_data')
    algorithm = settings.get('algorithms', 'algorithm')
    name = settings.get('algorithmName', algorithm)
    tuning = settings.getboolean('pipeline', 'hyperparameter_tuning')
    impute_missing = settings.getboolean('pipeline','impute_missing')
    significance_curve = settings.getboolean('pipeline','significance_curve')
    weighted = settings.getboolean('pipeline','weighted')

    logger.info('Running Algorithm: ' + name)

    logger.info('----------Starting Learning Pipeline -----------------')

    start_timer_pipeline = timeit.default_timer()

    higgs_data = pipeline.load(path,imputation=impute_missing)

    X_train = higgs_data.train_scaled
    Y_train = higgs_data.train_true_labels
    W_train = higgs_data.train_weights
    W_train_balanced = higgs_data.train_bweights

    X_test = higgs_data.test_scaled
    Y_test = higgs_data.test_true_labels
    W_test = higgs_data.test_weights
    W_test_balanced = higgs_data.test_bweights

    del higgs_data

    clf = ClassifierEngine(X_train, X_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, algorithm)

    if tuning:
        cv_scores, classifier, grid = pipeline.hyperparameter_tuning(clf, settings, weighted)
    else:
        classifier = pipeline.train_classifier_engine(clf, settings, weighted)

    try:
        selection_threshold = settings.getint('userParams', 'threshold')
    except ValueError:
        logger.warning('Percentile threshold for selection region suspended, defaulting to auto threshold')
        selection_threshold = settings.get('userParams', 'threshold')

    train_score, test_score, Y_test_predicted = pipeline.get_predictions(classifier, X_train, X_test, selection_threshold)

    selection_region = pipeline.derive_selection_region(classifier, X_test, Y_test_predicted)

    pipeline.performance_metrics(classifier, W_test, Y_test, Y_test_predicted, selection_region, selection_threshold, W_test_balanced)

    elapsed = pipeline.timer(start_timer_pipeline)

    logger.info('---------Finished Learning Pipeline----------')
    logger.info('*****Learning Pipeline took ' + str(elapsed) + ' seconds*****')

    if significance_curve:
        logger.info('-----------Generating AMS curve--------------')
        warnings.filterwarnings('ignore')
        curve, thresh = AMS.ams_curve(W_test, Y_test, test_score, train_score, classifier, new_fig=True, save=True, settings=settings)

    logger.removeHandler(out_handler)
    gc.collect()

#target = gzip.open('Pickled/clf_brf_opt.pklz','wb')
#pickle.dump(classifier.trained_classifier,target,-1)
#target.close()
#target = gzip.open('pickled_models/clf_38.pklz','rb')
#clf=pickle.load(target)
