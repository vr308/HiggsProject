# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:21:39 2016

@author: vr308
"""

from sklearn.preprocessing import StandardScaler
import gzip
import numpy as np
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
    return pd.DataFrame(scaled, columns=['cake'],index=df.index)

def scale_only(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.cake)
    return pd.DataFrame(scaled, columns=['cake'],index=df.index)

hd = HiggsData(path=settings.get('paths','path_data'), imputation=True)
test_cake = pd.read_csv('/local/data/public/vr308/higgsFudge_test.csv',';',header=0)
train_cake = pd.read_csv('/local/data/public/vr308/higgsFudge_train.csv',';',header=0)
new_cake = pd.read_csv('/local/data/public/vr308/higgsCakeNew.csv',';', header=0)

cake = pd.concat(objs=(train_cake,test_cake),axis=0)
cake.index = cake['EventId']
cake = log_transform_scale(cake)

new_cake.index = new_cake['EventId']
new_cake = scale_only(new_cake)

X_train = hd.train_scaled
Y_train = hd.train_true_labels
W_train = hd.train_weights
W_train_balanced = hd.train_bweights

X_test = hd.test_scaled
Y_test = hd.test_true_labels
W_test = hd.test_weights
W_test_balanced = hd.test_bweights

new_cake_train = pd.merge(X_train,new_cake, how='inner', left_index=True, right_index=True)
new_cake_test = pd.merge(X_test, new_cake, how='inner', left_index=True, right_index=True)

cake_train = pd.merge(X_train,cake, how='inner', left_index=True, right_index=True)
cake_test = pd.merge(X_test, cake, how='inner', left_index=True, right_index=True)

clf_cake = ClassifierEngine(cake_train, cake_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, 'BXT')
clf_new_cake = ClassifierEngine(new_cake_train, new_cake_test, Y_train, Y_test, W_train, W_test, W_train_balanced, W_test_balanced, 'BXT')

classifier_cake = pipeline.train_classifier_engine(clf_cake, settings, True)
classifier_new_cake = pipeline.train_classifier_engine(clf_new_cake, settings, True)

target = gzip.open('../Pickled/clf_bxt_opt.pklz','rb')
classifier=pickle.load(target)
target.close()

selection_threshold = settings.getint('userParams', 'threshold')

train_score, test_score, Y_test_predicted = pipeline.get_predictions(classifier, X_train, X_test, selection_threshold)
train_score_cake, test_score_cake, Y_test_predicted_cake = pipeline.get_predictions(classifier_cake, cake_train, cake_test, selection_threshold)
train_score_ncake, test_score_ncake, Y_test_predicted_ncake = pipeline.get_predictions(classifier_new_cake, new_cake_train, new_cake_test, selection_threshold)

selection_region_cake = pipeline.derive_selection_region(classifier_cake, cake_test, Y_test_predicted_cake)
selection_region_new_cake = pipeline.derive_selection_region(classifier_new_cake, new_cake_test, Y_test_predicted_ncake)
selection_region = pipeline.derive_selection_region(classifier, X_test, Y_test_predicted)

curve, thresh = AMS.ams_curve(W_test, Y_test, test_score, train_score, classifier, new_fig=True, save=False, settings=settings)
curve_cake, thresh = AMS.ams_curve(W_test, Y_test, test_score_cake, train_score_cake, classifier_cake, new_fig=False, save=True, settings=settings)
curve_ncake, thresh = AMS.ams_curve(W_test, Y_test, test_score_ncake, train_score_ncake, classifier_new_cake, new_fig=False, save=False, settings=settings)

target = gzip.open('Pickled/clf_new_cake_opt.pklz','wb')
pickle.dump(classifier_new_cake.trained_classifier,target,-1)
target.close()

