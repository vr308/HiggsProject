# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:40:02 2016

@author: vr308
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import gzip
import pickle
import ConfigParser
from higgs_data import HiggsData
from classifier_engine import ClassifierEngine
from discovery_significance import AMS
import pipeline
import pandas as pd
settings = ConfigParser.ConfigParser()
settings.read('settings.ini')

def log_transform_scale(df):
    df['cake'] = np.log(1 + df['cake'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.cake)
    return pd.DataFrame(scaled, columns=['cake'], index=df.index)

hd = HiggsData(path=settings.get('paths','path_data'), imputation=True)
test_cake = pd.read_csv('/local/data/public/vr308/higgsFudge_test.csv',';',header=0)
train_cake = pd.read_csv('/local/data/public/vr308/higgsFudge_train.csv',';',header=0)
new_cake = pd.read_csv('/local/data/public/vr308/higgsCakeNew.csv',';', header=0)

cake = pd.concat(objs=(train_cake,test_cake),axis=0)
cake.index = cake['EventId']
cake = log_transform_scale(cake)

X_train = hd.train_scaled
Y_train = hd.train_true_labels
W_train = hd.train_weights
W_train_balanced = hd.train_bweights

X_test = hd.test_scaled
Y_test = hd.test_true_labels
W_test = hd.test_weights
W_test_balanced = hd.test_bweights

cake_train = pd.merge(X_train,cake, how='inner', left_index=True, right_index=True)
cake_test = pd.merge(X_test, cake, how='inner', left_index=True, right_index=True)

clf = ClassifierEngine(cake_train, cake_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, 'BXT')
classifier_cake = pipeline.train_classifier_engine(clf, settings, True)

target = gzip.open('Pickled/clf_bxt_opt.pklz','rb')
classifier=pickle.load(target)
target.close()

selection_threshold = settings.getint('userParams', 'threshold')

train_score, test_score, Y_test_predicted = pipeline.get_predictions(classifier, X_train, X_test, selection_threshold)
train_score_cake, test_score_cake, Y_test_predicted_cake = pipeline.get_predictions(classifier_cake, cake_train, cake_test, selection_threshold)

selection_region_cake = pipeline.derive_selection_region(classifier_cake, cake_test, Y_test_predicted_cake)
selection_region = pipeline.derive_selection_region(classifier, X_test, Y_test_predicted)

curve, thresh = AMS.ams_curve(W_test, Y_test, test_score, train_score, classifier, new_fig=True, save=True, settings=settings)
curve_cake, thresh = AMS.ams_curve(W_test, Y_test, test_score_cake, train_score_cake, classifier_cake, new_fig=False, save=True, settings=settings)

#plt.figure()
#plt.hist(test_score[Y_test == -1], bins=100, histtype='stepfilled', alpha=0.4)
#plt.hist(test_score[Y_test == 1], bins=100, histtype='stepfilled', alpha=0.4)
#
#plt.figure()
#plt.hist(test_score_cake[Y_test == -1], bins=100, histtype='stepfilled', alpha=0.4)
#plt.hist(test_score_cake[Y_test == 1], bins=100, histtype='stepfilled', alpha=0.4)

#train_cake = cake[cake.index <= 350000]
#test_cake = cake[cake.index > 350000]
#
#train_labels = Y_train[Y_train.index <= 350000]
#test_labels = pd.concat([Y_test[Y_test.index > 350000], Y_train[Y_train.index > 350000]])
#
#plt.figure()
#plt.hist(train_cake['cake'][train_labels == -1].values, bins=100, histtype='stepfilled',alpha=0.4)
#plt.hist(train_cake['cake'][train_labels == 1].values, bins=100, histtype='stepfilled',alpha=0.4, color='r')
#
#plt.figure()
#plt.hist(test_cake['cake'][test_labels == -1].values, bins=100, histtype='stepfilled',alpha=0.4)
#plt.hist(test_cake['cake'][test_labels == 1].values, bins=100, histtype='stepfilled',alpha=0.4, color='r')
#
