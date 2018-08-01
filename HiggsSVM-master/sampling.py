# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 03:30:29 2016

@author: vr308
"""

import random
import numpy as np
import preprocessing 

def uniform_sampling(df,size):
    
    index = random.sample(df.index,size)
    return df.ix[index]
    
    
def choice_sampling(df,sigma):
    
    df_features = preprocessing.get_features(df)   
    index = []
    cols = df_features.columns 
    for i in cols :
        sd = np.std(df[i])
        mean = np.mean(df[i])
        set1 = df.index[df[i] < (mean + sigma*sd)]
        set2 = df.index[df[i] < (mean - sigma*sd)]
        set3 = set(set1-set2)
        index.append(set3)
    s = set.intersection(*index)
    df_sample = df.ix[s]
    return df_sample
    
def get_training_sample(train,sample_type,normalize):
    
    if(normalize):
         df_norm = preprocessing.normalize(train)[0]
         train = df_norm
         
    if sample_type == 'uniform':
        uni_sample = uniform_sampling(train,17000)
        uni_sample.index = np.arange(0,len(uni_sample))
        return uni_sample
    else:
        choice_sample = choice_sampling(train,1.6)
        
        # Selecting  10000 samples from the choice sample
        choice_sample = uniform_sampling(choice_sample,17000)
        choice_sample.index = np.arange(0,len(choice_sample))
        return choice_sample