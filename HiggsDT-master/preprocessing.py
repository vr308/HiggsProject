# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 03:07:37 2016

@author: vr308
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer



def load_data(path):
    
    df = pd.read_csv(path  + 'trainingForRelease.csv')
    df['Label'] =  map(lambda x : 1 if x == 's' else 0,df.Label)
    return df
    
def drop_features(df):
    
    cols = df.columns
    phi = [cols for cols in cols if cols.endswith('phi')]
    jetsPt = [cols for cols in cols if (cols.endswith('pt') & cols.startswith('PRI_jet'))]
    phi.extend(jetsPt)
    phi.extend('B')
    return df.drop(phi,axis=1)


def get_features(df):
    
    return df.drop(['Label','Weight','EventId'],axis=1)
    
def pop_features(df):
    
    cols = ['Label','Weight','EventId']
    return df.drop(cols,axis=1), df[cols]
    
def drop_missing_values(df):   
    
    cols = df.columns
    cols = cols.drop(['Label','Weight','EventId'])
    df['PRI_jet_num'] = df['PRI_jet_num'].astype(float)
    for i in cols :
        Id =  df[df[i] == -999.0].index
        df[i][df.index.isin(Id)] = np.NaN
    return df.dropna(how='any') 
 
   
def impute_missing_values(df):
    
    imp = Imputer(missing_values = -999.0, strategy='mean', verbose=0)
    df_imp = imp.fit_transform(df)
    return pd.DataFrame(df_imp,columns=df.columns)

def train_test_split(df,perc):
    
    r = np.random.rand(df.shape[0])
    
    train = df[r < perc]
    test = df[r >= perc]
    
    train.index = np.arange(0,len(train))    
    test.index = np.arange(0,len(test))    

    return train, test
    

 