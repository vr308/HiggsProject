# -*- coding: utf-8 -*-
"""
This script is a classification engine class that enables a user to run various
tree ensembles by choosing the set-up through the settings file.

@author: vr308
"""

import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_val
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from performance import PerformanceReports
import sklearn.ensemble as ensemble
import sklearn.tree as tree

class ClassifierEngine:

    _classification_algortihms = {
                    'DT': tree.DecisionTreeClassifier(),
                    'BDT': ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=10, max_features=15, min_samples_split = 150, min_samples_leaf = 150)),
                    'RF': ensemble.RandomForestClassifier(oob_score=True,warm_start=False,n_jobs=4,verbose=1),
                    'ET': ensemble.ExtraTreesClassifier(bootstrap=False,oob_score=False,n_jobs=4,verbose=1),
                    'BRF': ensemble.AdaBoostClassifier(base_estimator=ensemble.RandomForestClassifier(n_estimators=100, max_depth=13, max_features=20, min_samples_leaf=150, min_samples_split=150, n_jobs=4, verbose=1)),
                    'BXT': ensemble.AdaBoostClassifier(base_estimator=ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=13, min_samples_split=150, min_samples_leaf=150, max_features=20, n_jobs=4, verbose=1))}

    def __init__(self, X_train, X_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, algorithm):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.W_train = W_train
        self.W_test = W_test
        self.W_train_balanced = W_train_balanced
        self.W_test_balanced = W_test_balanced
        self.algorithm = algorithm
        self.untrained_classifier = self.get_classifier_model()
        self.param_dict = self.untrained_classifier.get_params()
        self.param_ranges = self.get_parameter_ranges()
        self.best_params = None
        self.trained_classifier = None
        self.train_posterior = None
        self.test_posterior = None

    def get_classifier_model(self):
        return self._classification_algortihms.get(self.algorithm)

    def get_parameter_ranges(self):
        param_ranges = {}
        if self.algorithm == 'DT':
            param_ranges.update(
                {
                'criterion'   : ['gini','entropy'],
                'max_features': [10, 15, 25],
                'max_depth'   : [5, 10, 15, 30],
                'min_samples_split' : [100, 300, 1000],
                'min_samples_leaf'  : [100, 300, 1000]
                })
        if self.algorithm == 'BDT':
            param_ranges.update(
                {
                 'base_estimator':[],
                 'n_estimators': [1, 10],
                 'learning_rate': [0.75, 1]
                })
        if self.algorithm == 'RF':
            param_ranges.update(
                {'n_estimators': [10, 25, 50, 100, 200],
                 'criterion': ['gini'],
                 'max_features': [15, 25],
                 'max_depth': [10, 12, 18],
                 'min_samples_split': [150],
                 'min_samples_leaf': [150]
                })
        if self.algorithm == 'ET':
            param_ranges.update(
                {'n_estimators': [10, 25, 50, 100, 200],
                 'criterion': ['gini'],
                 'max_features': [15, 25],
                 'max_depth': [10, 12, 18],
                 'min_samples_split': [150],
                 'min_samples_leaf': [150]
                })
        if self.algorithm == 'BRF':
            param_ranges.update(
                {
                'base_estimator': [],
                'n_estimators': [3, 5, 10, 20],
                'learning_rate': [0.73]
                })
        if self.algorithm == 'BXT':
            param_ranges.update(
                {
                'base_estimator': [],
                'n_estimators': [3, 5, 10, 20],
                'learning_rate': [0.73]
                })
        return param_ranges

    def grid_search(self, weighted, settings):
        if weighted:
            fit_params = {'sample_weight': self.W_train}
            filename_tag = 'weighted'
        else:
            fit_params = {}
            filename_tag = 'unweighted'
        param_grid = self.param_ranges
        if param_grid.has_key('base_estimator'):
            param_grid.pop('base_estimator')
        custom_error = make_scorer(PerformanceReports.balanced_classification_error, balanced_weights=self.W_train_balanced, greater_is_better=False)
        cross_val_object = cross_val.StratifiedKFold(np.array(self.Y_train),
                                     n_folds=3, shuffle=True, random_state=42)
        print 'Untrained Classifier: ' + str(self.untrained_classifier)
        print 'Parameter Grid: ' + str(param_grid)
        grid = GridSearchCV(self.untrained_classifier,
                             param_grid=param_grid,
                             cv=cross_val_object,
                             fit_params=fit_params,
                             scoring=custom_error,
                             verbose=True,
                             n_jobs=1)
        grid.fit(self.X_train, self.Y_train)

        self.best_params = grid.best_params_
        self.trained_classifier = grid.best_estimator_

        # grid_scores_ contains parameter settings and scores
        # We extract just the scores

        filename = 'Results/grid_scores_' + self.algorithm + '_' + filename_tag + '.txt'
        csv_filename = 'Results/grid_scores_' + self.algorithm + '_' + filename_tag + '.csv'
        print ('Writing output to file: ' + filename + '\n')
        mean_scores = [round(-1*x[1],4) for x in grid.grid_scores_]
        std_scores =[round(np.std(x[2]),4) for x in grid.grid_scores_]

        params  = pd.DataFrame(columns=self.param_ranges.keys())
        for i in xrange(len(mean_scores)):
            temp = pd.Series(grid.grid_scores_[i][0])
            params = pd.concat([params,pd.DataFrame(temp).T])
        params['test_error'] = mean_scores
        params['std_error'] = std_scores
        params.index = xrange(len(params))
        params.to_csv(csv_filename,sep=',', header=True, index=False)
        target = open(filename, 'w')
        target.write('Grid search for hyperparameters of ' + settings.get('algorithmName',self.algorithm) + '| Optimizing metric: Balanced Classification Error' + '\n' + '\n')
        target.write(params.to_string(index=False))
        target.close()
        return params, self, grid

    def train_classifier(self, weighted, args=None):
        if args is not None:
            for i in self.param_ranges.keys():
                if args.has_key(i):
                    self.param_dict[i] = args.get(i)
                    setattr(self.untrained_classifier, i, args.get(i))
            self.best_params = args
        if weighted:
            self.trained_classifier = self.untrained_classifier.fit(self.X_train, self.Y_train, self.W_train)
        else:
            self.trained_classifier = self.untrained_classifier.fit(self.X_train, self.Y_train)

    def get_decision_scores(self, data):
        try:
            decision_score = self.trained_classifier.decision_function(data)
        except AttributeError:
            decision_score = self.trained_classifier.predict_proba(data)[:, 1]

        return pd.Series(decision_score, index=data.index)

    def posterior_probability(self):

        self.test_posterior = self.trained_classifier.predict_proba(self.X_test)
        self.train_posterior = self.trained_classifier.predict_proba(self.X_train)

    def get_predicted_labels(self, X_test, test_score, cutoff):

        if cutoff == 'auto':
            predicted_labels = self.trained_classifier.predict(X_test)
        else:
            predicted_labels = pd.Series(map(lambda x: 1 if x > cutoff else -1, test_score), index=X_test.index)
        return predicted_labels

    def get_score_cutoff_for_threshold(self, train_score, selection_threshold):

        if selection_threshold == 'auto':
            return 'auto'
        else:
            return np.percentile(train_score, selection_threshold)

    def get_selection_events(self, X_test, Y_test_predicted):

        selection_events = X_test[Y_test_predicted == 1]
        selection_weights = self.W_test.ix[selection_events.index]
        selection_labels = self.Y_test.ix[selection_events.index]
        return selection_events, selection_weights, selection_labels

