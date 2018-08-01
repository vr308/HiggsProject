# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 01:57:08 2016

@author: vr308
"""
import numpy as np

def ams_compute(b,s):
    
   return  np.math.sqrt (2.*( (s + b + 10.)*np.math.log(1.+s/(b+10.))-s))
   
def ams_score(model,X_train,X_test,Y_train,Y_test,W_train,W_test,cut): 
    
    prob_train_score = model.decision_function(X_train) 
    prob_test_score = model.decision_function(X_test) 

    # Experience shows that choosing the top 15% as signal gives a good AMS score.
    
    pcut = np.percentile(prob_train_score,cut)
    
    # Yhat is the vector that governs the selection region
    
    Yhat_train = prob_train_score > pcut 
    Yhat_test = prob_test_score > pcut
    
    # Unbiased estimator of expected signal and background events obtained 
    # by summing the unnormalized importance weights over each class 
    # in the full dataset of 250,000 events
    
    N_s = 691.98
    N_b = 410999.84
    
    # Computing sum of weights for signal and background classes in the training and testing subset 
    
    # Y_train -> class label 1/0 for the training set
    # Y_test -> class label 1/0 for the testing set
    # W_train -> old weights in the training set
    # W_test -> old weights in the test set
    
    sum_weight_S_train    = sum ( W_train * (Y_train == 1.0))
    sum_weight_B_train    = sum ( W_train * (Y_train == 0.0))
    sum_weight_S_test     = sum ( W_test  * (Y_test  == 1.0))
    sum_weight_B_test     = sum ( W_test  * (Y_test  == 0.0))

    # Renormalizing weights (as per eq. 31) such that sum of weights in each class in 
    # the training and test subsets are equal to N_s and N_b respectively
    
    weight_S_new_train   =  W_train * (Y_train == 1.0) * (N_s / sum_weight_S_train)
    weight_B_new_train   =  W_train * (Y_train == 0.0) * (N_b / sum_weight_B_train)
    weight_S_new_test    =  W_test  * (Y_test  == 1.0) * (N_s / sum_weight_S_test)
    weight_B_new_test    =  W_test  * (Y_test  == 0.0) * (N_b / sum_weight_B_test)
    
    # Calculating sum of weights for signal and background events in the selection 
    # region defined by vector Yhat. Yhat = 1.0 for all events in the selection region 
    
    s_train              = sum ( weight_S_new_train * (Yhat_train == 1.0)) 
    b_train              = sum ( weight_B_new_train * (Yhat_train == 1.0))
    s_test               = sum ( weight_S_new_test  * (Yhat_test  == 1.0))
    b_test               = sum ( weight_B_new_test  * (Yhat_test  == 1.0))    
    
    ams_train = ams_compute(b_train,s_train)
    ams_test = ams_compute(b_test,s_test)
    
    print 'AMS Test : ' + str(ams_test)
    
    return ams_train, ams_test




























    
