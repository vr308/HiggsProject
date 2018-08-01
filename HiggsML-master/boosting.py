# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:05:18 2016

@author: vr308
"""


from ..base import is_regressor
import ConfigParser
from .forest import BaseForest
from ..utils import  check_X_y
from ..tree._tree import DTYPE
from numpy.core.umath_tests import inner1d
from ..tree.tree import BaseDecisionTree
import numpy as np

settings = ConfigParser.ConfigParser()
settings.read('settings.ini')
algorithm = settings.get('algorithms', 'algorithm')
selection_threshold = settings.get('userParams', 'threshold')

#higgs_data = pipeline.load(path)
#
#X_train = higgs_data.train_scaled
#Y_train = higgs_data.train_true_labels
#W_train = higgs_data.train_weights
#W_train_balanced = higgs_data.train_bweights
#
#X_test = higgs_data.test_scaled
#Y_test = higgs_data.test_true_labels
#W_test = higgs_data.test_weights
#W_test_balanced = higgs_data.test_bweights
#
#del higgs_data

import sklearn.ensemble as ensemble

class MetaB (ensemble.AdaBoostClassifier):

    def __init__(self):

        super(MetaB, self).__init__()

    def _boost_bifurcate_real(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

        for iboost in range(self.n_estimators):
            # Boosting step
#            sample_weight, estimator_weight, estimator_error = self._boost(
#                iboost,
#                X, y,
#                sample_weight)
            sample_weight, estimator_weight, estimator_error = self._boost_bifurcate_real(
                iboost,
                X, y,
                sample_weight)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

            return self































#    def boost (M, base_learner):
#
#
#
#    def bag (n_trees, clf):
#
#    randomTrees = pipeline.train_classifier_engine(clf,settings, weighted=False)
#
#    test_score = clf.get_decision_scores(clf.X_test)
#    train_score = clf.get_decision_scores(clf.X_train)
#
#    cutoff = clf.get_score_cutoff_for_threshold(train_score, selection_threshold)
#
#    Y_test_predicted = clf.get_predicted_labels(X_test, test_score, cutoff)
#
#    selection_region = pipeline.derive_selection_region(clf, X_test, Y_test_predicted)
#
#
#
#def bifurcate(base_estimator):
#
#    # Bifurcate training data into ones you need to boost and ones you do not
#
#
#
#
## Model
#
#clf1 = AdaBoostClassifier(n_estimators=20,
#                                  learning_rate=0.72,
#                                  base_estimator=ExtraTreesClassifier(
#                                  n_estimators=100,
#                                  criterion='gini',
#                                  max_depth=13,
#                                  max_features = 17,
#                                  min_samples_split=100,
#                                  min_samples_leaf = 100,
#                                  n_jobs=4,
#                                  verbose=1))
#
#clf1.fit(X_train,Y_train,W_train)
#
#clf_base = ClassifierEngine(X_train, X_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, '')
