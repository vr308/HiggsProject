# -*- coding: utf-8 -*-
"""
This module provides methods for loading, preprocessing and scaling
the dataset.

@author: vr308
"""

import pandas as pd
import numpy as np
import angles as angles

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler


class Preprocessing:

    non_features = ['Label', 'Weight', 'EventId', 'KaggleSet', 'KaggleWeight']

    @classmethod
    def der_features(cls, df):
        cols = df.columns
        sub = [cols for cols in df.columns if cols.startswith('DER')]
        return df[sub]

    @classmethod
    def pri_features(cls, df):
        cols = df.columns
        sub = [cols for cols in df.columns if cols.startswith('PRI')]
        return df[sub]

    @classmethod
    def get_features(cls, df):
        drop_cols = df.columns & Preprocessing.non_features
        return df.drop(drop_cols, axis=1)

    @classmethod
    def pop_features(cls, df):
        return df.drop(Preprocessing.non_features, axis=1), df[Preprocessing.non_features]

    @classmethod
    def replace_missing_values(cls, df, value):
        cols = Preprocessing.get_features(df).columns
        df['PRI_jet_num'] = df['PRI_jet_num'].astype(float)
        df['PRI_jet_all_pt'] = np.abs(df['PRI_jet_all_pt'])
        for i in cols:
            Id = df[df[i] == -999.0].index
            df[i][df.index.isin(Id)] = value
            return df

    @classmethod
    def remove_missing_values(cls, df, value):
        d_replace = Preprocessing.replace_missing_values(df, np.NaN)
        return d_replace.dropna(how='any', axis=0)

    @classmethod
    def impute_missing_values(cls, df):
        imp = Imputer(missing_values=-999.0, strategy='median', verbose=0)
        features, non = Preprocessing.pop_features(df)
        df_imp = pd.DataFrame(imp.fit_transform(features), columns=features.columns, index=features.index)
        return pd.merge(df_imp, non, left_index=True, right_index=True)

    @classmethod
    def add_mass_features(cls, df):
        tau_momentum_vec = Preprocessing.compute_four_momentum_vectors(df['PRI_tau_pt'], df['PRI_tau_phi'], df['PRI_tau_eta'])
        lep_momentum_vec = Preprocessing.compute_four_momentum_vectors(df['PRI_lep_pt'], df['PRI_lep_phi'], df['PRI_lep_eta'])
        jet_leading_momentum_vec = Preprocessing.compute_four_momentum_vectors(df['PRI_jet_leading_pt'],
                 df['PRI_jet_leading_phi'], df['PRI_jet_leading_eta'])
        jet_subleading_momentum_vec = Preprocessing.compute_four_momentum_vectors(df['PRI_jet_subleading_pt'],
                 df['PRI_jet_subleading_phi'], df['PRI_jet_subleading_eta'])
        df['DER_mass_invariant_tau_lep'] = Preprocessing.compute_mass_invariant(tau_momentum_vec, lep_momentum_vec)
        df['DER_mass_invariant_tau_jet1'] = Preprocessing.compute_mass_invariant(tau_momentum_vec, jet_leading_momentum_vec)
        df['DER_mass_invariant_tau_jet2'] = Preprocessing.compute_mass_invariant(tau_momentum_vec, jet_subleading_momentum_vec)
        df['DER_mass_transverse_tau_jet1'] = Preprocessing.compute_mass_transverse(tau_momentum_vec, jet_leading_momentum_vec)
        df['DER_mass_transverse_tau_jet2'] = Preprocessing.compute_mass_transverse(tau_momentum_vec, jet_subleading_momentum_vec)
        return df

    @classmethod
    def compute_mass_invariant(cls, mt_a, mt_b):

        mt_a_sq = (Preprocessing.square_1d(mt_a[0]), Preprocessing.square_1d(mt_a[1]), Preprocessing.square_1d(mt_a[2]))
        mt_b_sq = (Preprocessing.square_1d(mt_b[0]), Preprocessing.square_1d(mt_b[1]), Preprocessing.square_1d(mt_b[2]))
        sum_sq_a = Preprocessing.sqrt_sum_sq_3d(mt_a_sq)
        sum_sq_b = Preprocessing.sqrt_sum_sq_3d(mt_b_sq)
        sum_ab = Preprocessing.square_sum_1d(sum_sq_a, sum_sq_b)
        sq_sum_x = Preprocessing.square_sum_1d(mt_a[0], mt_b[0])
        sq_sum_y = Preprocessing.square_sum_1d(mt_a[1], mt_b[1])
        sq_sum_z = Preprocessing.square_sum_1d(mt_a[2], mt_b[2])
        sq_sum = np.where(sum_ab != -999.0, np.abs(sum_ab - sq_sum_x - sq_sum_y - sq_sum_z), -999.0)
        return Preprocessing.sqrt_1d(sq_sum)

    @classmethod
    def square_1d(cls, mt):
        return np.where(mt != -999.0, np.square(mt), -999.0)

    @classmethod
    def sum_1d(cls, mt_a, mt_b):
        return np.where(mt_b != -999.0, mt_a + mt_b, -999.0)

    @classmethod
    def sqrt_1d(cls, mt):
        return np.where(mt != -999.0, np.sqrt(mt), -999.0)

    @classmethod
    def square_sum_1d(cls, mt_a, mt_b):
        return Preprocessing.square_1d(Preprocessing.sum_1d(mt_a, mt_b))

    @classmethod
    def sqrt_sum_sq_3d(cls, mt_squared):
        return np.where(mt_squared[0] != -999.0, np.sqrt(np.sum(mt_squared, axis=0)), -999.0)

    @classmethod
    def log_transform_1d(cls, feature):
        return np.where(feature != -999.0, np.log(1 + feature), -999.0)

    @classmethod
    def compute_mass_transverse(cls, mt_a, mt_b):
        sum_pt_a = Preprocessing.sqrt_1d(Preprocessing.sum_1d(Preprocessing.square_1d(mt_a[0]), Preprocessing.square_1d(mt_a[1])))
        sum_pt_b = Preprocessing.sqrt_1d(Preprocessing.sum_1d(Preprocessing.square_1d(mt_b[0]), Preprocessing.square_1d(mt_b[1])))
        sum_pt = Preprocessing.square_sum_1d(sum_pt_a, sum_pt_b)
        sq_sum_x = Preprocessing.square_sum_1d(mt_a[0], mt_b[0])
        sq_sum_y = Preprocessing.square_sum_1d(mt_a[1], mt_b[1])
        sq_sum = np.where(sum_pt != -999.0, np.abs(sum_pt - sq_sum_x - sq_sum_y), -999.0)
        return Preprocessing.sqrt_1d(sq_sum)

    @classmethod
    def compute_four_momentum_vectors(cls, pt, phi, eta):
        px = np.where(np.logical_and(pt != -999.0, phi != -999.0), np.multiply(pt, np.cos(phi)), -999.0)
        py = np.where(np.logical_and(pt != -999.0, phi != -999.0), np.multiply(pt, np.sin(phi)), -999.0)
        pz = np.where(np.logical_and(pt != -999.0, eta != -999.0), np.multiply(pt, np.sinh(eta)), -999.0)
        return (px, py, pz)

    @classmethod
    def azhimuth_angles(cls, df):
        tau_lep = Preprocessing.normalize_angles(df['PRI_tau_phi'] - df['PRI_lep_phi'])
        tau_met = Preprocessing.normalize_angles(df['PRI_tau_phi'] - df['PRI_met_phi'])
        lep_met = Preprocessing.normalize_angles(df['PRI_lep_phi'] - df['PRI_met_phi'])
        nd = np.c_[tau_lep, tau_met, lep_met]
        df['PRI_radian1'] = np.amin(nd, axis=1)
        df['PRI_radian2'] = np.amin(nd[:, (1, 2)], axis=1)
        df['PRI_radian3'] = np.amin(nd[:, (0, 1)], axis=1)
        df['PRI_radian4'] = lep_met
        return df.drop(labels=['PRI_tau_phi', 'PRI_lep_phi', 'PRI_met_phi', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], axis=1)

    @classmethod
    def flip_eta(cls, df):
        df['PRI_lep_eta'] = np.where(df['PRI_tau_eta'] < 0, np.multiply(-1, df['PRI_lep_eta']), df['PRI_lep_eta'])
        df['PRI_jet_leading_eta'] = np.where(np.logical_and(df['PRI_tau_eta'] < 0, df['PRI_jet_leading_eta'] != -999.0), np.multiply(-1, df['PRI_jet_leading_eta']), df['PRI_jet_leading_eta'])
        df['PRI_jet_subleading_eta'] = np.where(np.logical_and(df['PRI_tau_eta'] < 0, df['PRI_jet_subleading_eta'] != -999.0),  np.multiply(-1, df['PRI_jet_subleading_eta']), df['PRI_jet_leading_eta'])
        df['PRI_tau_eta'] = np.where(df['PRI_tau_eta'] < 0, np.multiply(-1, df['PRI_tau_eta']), df['PRI_tau_eta'])
        return df

    @classmethod
    def log_transform(cls, df):

        pos_columns = ['DER_mass_MMC',
                       'DER_mass_transverse_met_lep',
                       'DER_mass_vis',
                       'DER_pt_h',
                       'DER_deltaeta_jet_jet',
                       'DER_mass_jet_jet',
                       'DER_deltar_tau_lep',
                       'DER_pt_tot',
                       'DER_sum_pt',
                       'DER_pt_ratio_lep_tau',
                       'DER_lep_eta_centrality',
                       'DER_mass_invariant_tau_lep',
                       'DER_mass_invariant_tau_jet1',
                       'DER_mass_invariant_tau_jet2',
                       'DER_mass_transverse_tau_jet1',
                       'DER_mass_transverse_tau_jet2',
                       'PRI_tau_pt',
                       'PRI_lep_pt',
                       'PRI_met',
                       'PRI_met_sumet',
                       'PRI_jet_leading_pt',
                       'PRI_jet_subleading_pt',
                       'PRI_jet_all_pt']

        df_inv_log_cols = df[pos_columns].apply(Preprocessing.log_transform_1d, axis=0)
        #df_inv_log_cols = np.log(1 + df[pos_columns])
        return pd.merge(df[df.columns.diff(pos_columns)], df_inv_log_cols, left_index=True, right_index=True)

    @classmethod
    def normalize_angles(cls, radian_vector):

        return np.asarray(map(lambda x: angles.d2r(angles.normalize(angles.r2d(x), -180, +180)), radian_vector))

    @classmethod
    def get_scaler(cls, df):

        #df = Preprocessing.impute_missing_values(df)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
        return scaled, scaler

    @classmethod
    def scale_inputs(cls, df, scaler):

        scaled = scaler.transform(df)
        scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
        return scaled

    @classmethod
    def generate_preprocessed_input(cls, df, imputation):

        df = Preprocessing.add_mass_features(df)
        df = Preprocessing.azhimuth_angles(df)
        df = Preprocessing.flip_eta(df)
        #df =  Preprocessing.replace_missing_values(df)
        df = Preprocessing.log_transform(df)
        if imputation:
            df = Preprocessing.impute_missing_values(df)
        return df[df.KaggleSet != 'u']