# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 01:57:08 2016

@author: vr308
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.stats.stats import nanmean
import tree_model

def ams_score(model,X_train,X_test,Y_train,Y_test,W_train,W_test,cut,probability_model): 
    
    if probability_model == 1:
        prob_train_score = tree_model.model_probability(model, X_train,Y_train) 
        prob_test_score = tree_model.model_probability(model, X_test,Y_test)
    else :
        prob_train_score = tree_model.calibrate_probability(model, X_train,Y_train) 
        prob_test_score = tree_model.calibrate_probability(model, X_test,Y_test)
    
    # A lot of successful models have shows that the AMS is maximized at threshold of 85%
    
    pcut = np.percentile(prob_train_score,cut)
    
    # The selection region 
    
    Yhat_train = prob_train_score >= pcut 
    Yhat_test = prob_test_score >= pcut
    
    # Unbiased estimator of expected signal and background events obtained 
    # by summing the unnormalized importance weights over each class in the  
    
    N_s = 691.98       # Unbiased estimator of number of signal events obtained from full dataset
    N_b = 410999.84    # Unbiased estimator of number of background events obtained from full dataset
    
    sum_of_weights_signal_train_set = sum(W_train*(np.asarray(Y_train)==1.0))
    sum_of_weights_background_train_set = sum(W_train*(np.asarray(Y_train)==0.0))
    sum_of_weights_signal_test_set = sum(W_test*(np.asarray(Y_test)==1.0))
    sum_of_weights_background_test_set = sum(W_test*(np.asarray(Y_test)==0.0))

    # To calculate the AMS, first get the true 
    # Scale the weights according to fraction of training data used

    TruePositive_train = W_train*(np.asarray(Y_train)==1.0)*(N_s/sum_of_weights_signal_train_set)
    FalsePositive_train = W_train*(np.asarray(Y_train)==0.0)*(N_b/sum_of_weights_background_train_set)
    TruePositive_test = W_test*(np.asarray(Y_test)==1.0)*(N_s/sum_of_weights_signal_test_set)
    FalsePositive_test = W_test*(np.asarray(Y_test)==0.0)*(N_b/sum_of_weights_background_test_set)
    
    s_train = sum ( TruePositive_train*(Yhat_train==1.0) ) 
    b_train = sum ( FalsePositive_train*(Yhat_train==1.0) )
    s_test = sum ( TruePositive_test*(Yhat_test==1.0) )
    b_test = sum ( FalsePositive_test*(Yhat_test==1.0) )    
    
    ams_train = ams_compute(b_train,s_train)
    ams_test = ams_compute(b_test,s_test)
    
    return ams_train, ams_test


def ams_compute(b,s):
    
   return  np.math.sqrt (2.*( (s + b)*np.math.log(1.+s/b)-s))
   

def average_ams(model,X_train,X_test, Y_train,Y_test, W_train,W_test,cuts,prob,bm):
    
    ams_train_list = []
    ams_test_list = []
    for i in cuts:
         ams_train, ams_test =  ams_score(model,X_train,X_test,Y_train,Y_test,W_train,W_test,i,prob,bm)
         ams_train_list.append(ams_train)
         ams_test_list.append(ams_test)
    return nanmean(ams_train_list), nanmean(ams_test_list)

def ams_curve(model,X_train,X_test,Y_train,Y_test,W_train,W_test,thresh,prob):
    
     ams_train_curve = []
     ams_test_curve = []
     for i in thresh:
         ams_train, ams_test =  ams_score(model,X_train,X_test,Y_train,Y_test,W_train,W_test,i,prob)
         ams_train_curve.append(ams_train)
         ams_test_curve.append(ams_test)
     return ams_train_curve,ams_test_curve

    
def plot_ams_curve(cuts,ams_curve,label):
     
     peak_ams = max(ams_curve)
     best_cut = cuts[ams_curve.index(max(ams_curve))]
     plt.figure()
     plt.grid()
     plt.xlim(min(cuts),max(cuts))
     plt.plot(cuts,ams_curve,'r+-',label='Test AMS')
     plt.axhline(y=peak_ams,label='Peak AMS = '+str(round(peak_ams,2)),color='black',linestyle='--')
     plt.axvline(x=best_cut,linestyle='--',label = 'Best threshold = ' + str(best_cut),color='black')
     plt.xlabel('Threshold (% Rejected)')
     plt.ylabel('AMS Score')
     plt.legend(loc=2)
     plt.title('AMS Curve ' + label) 
     
     
     
#def ams_score(model,X_train,X_test,Y_train,Y_test,W_train,W_test,cut,prob,bm): 
#    
#    if prob == 1:
#        prob_train_score = tree_model.model_probability(model, X_train,Y_train) 
#        prob_test_score = tree_model.model_probability(model, X_test,Y_test)
#    else :
#        prob_train_score = tree_model.calibrate_probability(model, X_train,Y_train,bm) 
#        prob_test_score = tree_model.calibrate_probability(model, X_test,Y_test,bm)
#        
# 
#    # Experience shows me that choosing the top 15% as signal gives a good AMS score.
#    pcut = np.percentile(prob_train_score,cut)
#    
#    # This are the final signal and background predictions
#    
#    Yhat_train = prob_train_score  > pcut 
#    Yhat_test = prob_test_score > pcut
#    
#    train_frac = len(Y_train)/250000.0
#    test_frac = len(Y_test)/250000.0
#     
#    # To calculate the AMS data, first get the true positives and true negatives
#    # Scale the weights according to fraction
#    
#    TruePositive_train = W_train*(np.asarray(Y_train)==1.0)*(1.0/train_frac)
#    TrueNegative_train = W_train*(np.asarray(Y_train)==0.0)*(1.0/train_frac)
#    TruePositive_valid = W_test*(np.asarray(Y_test)==1.0)*(1.0/test_frac)
#    TrueNegative_valid = W_test*(np.asarray(Y_test)==0.0)*(1.0/test_frac)
#       
#    # Counting the number of signals and background  
#    
#    s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
#    b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
#    s_test = sum ( TruePositive_valid*(Yhat_test==1.0) )
#    b_test = sum ( TrueNegative_valid*(Yhat_test==1.0) )    
#    
#    AMS_Train = ams_compute(b_train,s_train)
#    AMS_Test = ams_compute(b_test,s_test)
#    
#    #print AMS_Train
#    #print AMS_Test 
#    
#    return AMS_Train, AMS_Test