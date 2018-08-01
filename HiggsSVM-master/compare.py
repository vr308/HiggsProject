# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:35:16 2016

@author: vr308
"""

#os.chdir(os.getcwd()+ '/higgsSVM')

import numpy as np
import sys
import pandas as pd
import matplotlib.pylab as plt
import itertools

import preprocessing 
import sampling 
import cross_validation 
import discovery_significance


train_sample_type = 'uniform_sample' 
code_path = '/home/raid/vr308/workspace/Python/higgsSVM/'

def generate_raw_data():
    
    df = preprocessing.load_data(path=code_path +'/Data/')
    df = preprocessing.drop_features(df)
    df = preprocessing.drop_missing_values(df)
    df = preprocessing.normalize(df)[0]
    train, test = preprocessing.train_test_split(df,perc=0.80)    
    return train,test    
    
def generate_train_sample(train,train_sample_type):
    
    train_uniform = sampling.get_training_sample(train,sample_type='uniform',normalize=False)
    train_choice = sampling.get_training_sample(train,sample_type='choice',normalize=False)
        
    if (train_sample_type == 'choice_sample'):
            train_sample = train_choice  
    else:
            train_sample = train_uniform
    
    return train_sample
    
def average_ams(train,test,train_sample_type):
    
    ams_NA = []
    ams_A = []
    
    iterations = np.arange(1,4,1)
    
    for i in iterations:
        
        print 'Iteration ' + str(i) 
        
        train_sample = generate_train_sample(train,train_sample_type)
        
        X_train = preprocessing.get_features(train_sample)    
        Y_train = train_sample['Label']
        W_train = train_sample['Weight']
                
        #X_test = preprocessing.normalize(test)[0]
        #X_test = preprocessing.normalize_fit_test(preprocessing.normalize(train)[1],test)
        X_test = preprocessing.get_features(test)
        Y_test = test['Label']
        W_test = test['Weight']
        #W_test_balanced = test.pop('W_balanced')
        
        X_train_sub = X_train.drop(labels=['A'],axis=1)
        X_train_plus = X_train
        
        X_test_sub = X_test.drop(labels=['A'],axis=1)
        X_test_plus = X_test        
            
        modelA = cross_validation.fit_svm(X_train_plus,Y_train,'rbf', C=100, gamma=0.0056)
        modelNA = cross_validation.fit_svm(X_train_sub,Y_train,'rbf', C=100, gamma=0.0056)
        thresholds,ams_train_A,ams_test_A=discovery_significance.ams_curve(modelA,X_train_plus,
                                                                           X_test_plus,np.asarray(Y_train),np.asarray(Y_test),W_train,W_test)
        thresholds,ams_train_NA,ams_test_NA=discovery_significance.ams_curve(modelNA,X_train_sub,
                                                                        X_test_sub,np.asarray(Y_train),np.asarray(Y_test),W_train,W_test)  
        
        ams_NA.append(ams_test_NA)
        ams_A.append(ams_test_A)
        
    ams_NA_mean = pd.DataFrame(ams_NA).mean(axis=0)
    ams_A_mean = pd.DataFrame(ams_A).mean(axis=0)
    
    ams_NA_std = pd.DataFrame(ams_NA).std(axis=0)
    ams_A_std = pd.DataFrame(ams_A).std(axis=0)
    
    plt.figure()
    plt.grid()
    plt.plot(thresholds,ams_NA_mean,'bo',label='No A')
    plt.plot(thresholds,ams_A_mean,'ro',label='With A')
    plt.errorbar(thresholds, ams_NA_mean, yerr=ams_NA_std)
    plt.errorbar(thresholds, ams_A_mean, yerr=ams_A_std)
    plt.title('AMS Compare')
    plt.legend(loc=3)
    plt.savefig('Graphs/AMS_compare_' + train_sample_type,format='png')
    
    return ams_NA, ams_A
    
    
def ams_grid(train,test):

    C = [10,100,1000]
    gamma = np.linspace(0.0001,0.05,10)
    param_grid = list(itertools.product(C,gamma))
    
    train_sample = generate_train_sample(train,train_sample_type)
        
    X_train = preprocessing.get_features(train_sample)    
    Y_train = train_sample['Label']
    W_train = train_sample['Weight']
            
    X_test = preprocessing.normalize(test)[0]
    X_test = preprocessing.get_features(X_test)
    Y_test = test['Label']
    W_test = test['Weight']
    #W_test_balanced = test.pop('W_balanced')
    
    X_train_sub = X_train.drop(labels=['A'],axis=1)
    X_train_plus = X_train
    
    X_test_sub = X_test.drop(labels=['A'],axis=1)
    X_test_plus = X_test        
        
    ams_noA = []
    ams_A = []
    
    for i in param_grid:
        
        C = i[0]
        gamma = i[1]
        
        modelA = cross_validation.fit_svm(X_train_plus,Y_train,'rbf', C=C, gamma=gamma)
        modelNA = cross_validation.fit_svm(X_train_sub,Y_train,'rbf', C=C, gamma=gamma)
        
        thresholds,ams_train_A,ams_test_A=discovery_significance.ams_curve(modelA,X_train_plus,
                                                    X_test_plus,Y_train,Y_test,W_train,W_test)
        thresholds,ams_train_NA,ams_test_NA=discovery_significance.ams_curve(modelNA,X_train_sub,
                                                        X_test_sub,Y_train,Y_test,W_train,W_test)  
        
        ams_noA.append(ams_test_NA)
        ams_A.append(ams_test_A)
        
    return ams_noA, ams_A
                
if __name__ == "__main__":

   train,test = generate_raw_data()
   ams_NA,ams_A = average_ams(train,test,train_sample_type)