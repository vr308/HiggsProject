# -*- coding: utf-8 -*-
"""
This module provides the data set storage class.
The higgs data is read into an object of this class type.

@author: vr308

"""

# Python Standard Library imports

import numpy as np
import pandas as pd

# Project module imports

from preprocessing import Preprocessing


class HiggsData:

    def __init__(self,path, imputation):

        self.raw_input = HiggsData.load_data(path)
        self.processed_input = Preprocessing.generate_preprocessed_input(self.raw_input, imputation)
        self.background = HiggsData.get_background(self.raw_input)
        self.signal = HiggsData.get_signal(self.raw_input)
        self.N_b = np.sum(self.background.Weight)
        self.N_s = np.sum(self.signal.Weight)
        self.train, self.train_true_labels, self.train_weights, self.train_bweights = HiggsData.split_input(self.processed_input, 'train')
        self.test, self.test_true_labels, self.test_weights, self.test_bweights = HiggsData.split_input(self.processed_input, 'test')
        self.train_scaled, self.scaler = Preprocessing.get_scaler(self.train)
        self.test_scaled = Preprocessing.scale_inputs(self.test, self.scaler)

    @classmethod
    def load_data(cls,path):

        iterator = pd.read_csv(filepath_or_buffer=path, chunksize=250000)
        df = iterator.get_chunk()
        df['Label'] = map(lambda x: 1 if x == 's' else -1, df.Label)
        df.index = df['EventId']
        return df

    @classmethod
    def get_background(cls, df):

        return df[df.Label == -1]

    @classmethod
    def get_signal(cls, df):

        return df[df.Label == 1]

    @classmethod
    def get_sum_weights(cls, df):

        signal = HiggsData.get_signal(df)
        background = HiggsData.get_background(df)
        return np.sum(signal.Weight), np.sum(background.Weight)

    @classmethod
    def get_balanced_weight(cls, df):

        N_s, N_b = HiggsData.get_sum_weights(df)
        N_est = np.array(map(lambda x: N_s if x == 1 else N_b, df.Label))
        df['NormWeight'] = df['Weight'] * (0.5 * (1 / N_est))
        return df['NormWeight']

    @classmethod
    def split_input(cls, df, flag, scaler=None):

        if flag == 'train':
            sub = df[(df.KaggleSet == 't') | (df.KaggleSet == 'b')]
        elif flag == 'test':
            sub = df[df.KaggleSet == 'v']
        X = Preprocessing.get_features(sub)
        W_ = HiggsData.get_balanced_weight(sub)
        Y = sub.pop('Label')
        W = sub.pop('KaggleWeight')
        return X, Y, W, W_