# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 05:36:01 2016

@author: vr308
"""

import numpy as np 
import sklearn.metrics as metrics
import sys

import matplotlib.pylab as plt
import discovery_significance
import tree_model
import pandas as pd

code_path = sys.argv[1]

    
if __name__ == "__main__":

#-------------------------Preprocessing steps -------------------------------------    
    print 'Step 1 : Loading Data'
    train = pd.read_csv(code_path + '/Data/TRAIN.csv') 
    test = pd.read_csv(code_path + '/Data/TEST.csv') 
    
    print 'Step 2 : Enriching train and test sets'
    
    N_b = sum(train['Weight'][train['Label'] == 0])
    N_s = sum(train['Weight'][train['Label'] == 1])
    
    train['N_est'] = map(lambda x : N_s if x == 1 else N_b,train.Label)
    
    train['W_balanced'] = train['Weight']*(0.5*(1/train['N_est'])) 
    
    Y_train = train['Label']
    W_train = train['W_balanced']
    X_train = train.drop(['Label','Weight','EventId','W_balanced','N_est'], axis=1)
    
    N_b = sum(test['Weight'][test['Label'] == 0])
    N_s = sum(test['Weight'][test['Label'] == 1])
    
    test['N_est'] = map(lambda x : N_s if x == 1 else N_b,test.Label)  
    test['W_balanced'] = test['Weight']*(0.5*(1/test['N_est'])) 
    
    Y_test = list(test['Label'])
    W_test = test['Weight']
    X_test = test.drop(['Label','Weight','EventId','W_balanced','N_est'], axis=1)

#---------------------Baseline Performance metrics --------------------------------------------------------------
    
    print 'Computing metrics for baseline unoptimized tree'
    
    _base_tree_gini = tree_model.fit_tree(X_train, Y_train,W_train,'gini',100,1,2,None)
    _base_tree_entropy = tree_model.fit_tree(X_train, Y_train,W_train,'entropy',100,1,2,None)

    gini_prob = _base_tree_gini.predict_proba(X_test)[:,1]
    entropy_prob = _base_tree_entropy.predict_proba(X_test)[:,1]   
    
    fpr_p1,tpr_p1, thresh_p1 = metrics.roc_curve(Y_test,gini_prob,pos_label=1)
    fpr_p2,tpr_p2, thresh_p2 = metrics.roc_curve(Y_test,entropy_prob,pos_label=1)
    
    plt.figure()
    plt.grid()
    plt.plot(fpr_p1,tpr_p1,label='Gini')
    plt.plot([0,1],[0,1],color='black',linestyle='--')
    plt.plot(fpr_p1, tpr_p1,'bo')
    plt.plot(fpr_p2,tpr_p2,label='Entropy')
    plt.plot(fpr_p2,tpr_p2,'go')
    plt.xlabel('1 - Specificity or False Positive Rate')
    plt.ylabel('Sensitivity or True Positive Rate')
    plt.title('ROC curve for baseline CART tree using default parameters')
    plt.savefig('Graphs/baseline_DT_ROC.png')
    plt.legend(loc=2)

    print 'Writing out performance metrics for unoptimized CART trees'
    
    print 'AMS under Entropy :' + str(discovery_significance.ams_score(_base_tree_entropy,X_train,X_test,Y_train,
                                                                      Y_test,train['Weight'],W_test,85,1))
    
    print 'AMS under Gini :' + str(discovery_significance.ams_score(_base_tree_gini,X_train,X_test,Y_train,
                                                                      Y_test,train['Weight'],W_test,85,1))
    tree_model.balanced_classification_error(_base_tree_entropy,X_test,Y_test,test.W_balanced)    
    
    print(metrics.classification_report(Y_test, _base_tree_gini.predict(X_test)))
    print(metrics.classification_report(Y_test, _base_tree_entropy.predict(X_test)))    
    
    print 'ROC score for baseline tree with Gini crietrion ' + str(metrics.roc_auc_score(Y_test, _base_tree_gini.predict(X_test)))
    print 'ROC score for baseline tree with Entropy criterion ' + str(metrics.roc_auc_score(Y_test, _base_tree_entropy.predict(X_test)))

# ----------------------Optimized Criterion ---------------------------------#
    print 'Step 4 : Setting optimum parameters for CART trees'
    
    metric           = ['balanced_classification_error','roc_auc']
    max_features      = [10,None]
    min_samples_split = [40,200]
    min_samples_leaf  = [150,150]
    criterion         = ['entropy','gini']
    max_depth         = [None,50]
    
    print ' Step 4 : Fitting CART tree with 2 sets of optimal parameters '
    
    _tree_opt_bce = tree_model.fit_tree(X_train, Y_train,W_train,criterion[0],
                                       max_depth[0],
                                       min_samples_leaf[0],
                                       min_samples_split[0],
                                       max_features[0])
    
    _tree_opt_roc = tree_model.fit_tree(X_train, Y_train,W_train,criterion[1],
                                       max_depth[1],
                                       min_samples_leaf[1],
                                       min_samples_split[1],
                                       max_features[1])
    
    print  'Step 5 : Compute optimized metrics for fitted trees  '
    print 'Writing out performance metrics for optimized CART trees'

    roc_prob = _tree_opt_roc.predict_proba(X_test)[:,1]
    bce_prob = _tree_opt_bce.predict_proba(X_test)[:,1]   

    print 'ROC  under ROC optimized tree : '  + str(metrics.roc_auc_score(Y_test, roc_prob))  
    print 'ROC  under R(f) optimized tree : ' +  str(metrics.roc_auc_score(Y_test, bce_prob))    

    print(metrics.classification_report(Y_test, _tree_opt_bce.predict(X_test)))
    print(metrics.classification_report(Y_test, _tree_opt_roc.predict(X_test)))    
    
    roc_error = tree_model.balanced_classification_error(_tree_opt_bce,X_test,Y_test,test.W_balanced) 
    bce_error = tree_model.balanced_classification_error(_tree_opt_roc,X_test,Y_test,test.W_balanced)    
    
    print 'Balanced classification error under ROC optimal tree : ' + str(roc_error)
    print 'Balanced classification error under R(f) optimal tree :' + str(bce_error)
    
    ams_roc = discovery_significance.ams_score(_tree_opt_roc,X_train,X_test,Y_train,
                                                                      Y_test,train['Weight'],W_test,83,1)
    ams_bce = discovery_significance.ams_score(_tree_opt_bce,X_train,X_test,Y_train,
                                                                      Y_test,train['Weight'],W_test,83,1)
    print 'AMS under ROC optimal tree : ' + str(ams_roc)
    print 'AMS under R(f) optimal tree : ' + str(ams_bce)
    
    print ' Step 4 : Plot ROC curves under the 2 fitted trees' 
    
    fpr_p0,tpr_p0, thresh_p0 = metrics.roc_curve(Y_test,roc_prob,pos_label=1)
    fpr_p1,tpr_p1, thresh_p1 = metrics.roc_curve(Y_test,bce_prob,pos_label=1)
    
    plt.figure()
    plt.grid()
    plt.plot(fpr_p0,tpr_p0,label='ROC maximizing classifier')
    plt.plot([0,1],[0,1],color='black',linestyle='--')
    plt.plot(fpr_p1,tpr_p1,label='BCE minimizing classifier')
    plt.xlabel('1 - Specificity or False Positive Rate')
    plt.ylabel('Sensitivity or True Positive Rate')
    plt.title('ROC curves for CART tree using optimum parameters')
    plt.legend(loc='best')
    plt.savefig('Graphs/ROC_Optimum.png',bbox_inches='tight')
   
    print 'Step 5 : Plot AMS curves using the two optimizing sets of parameters over a whole range of thresholds'
    
    thresh = list(np.arange(10,95,0.5))
    ams_train_0, ams_test_0 = discovery_significance.ams_curve(_tree_opt_roc,X_train,X_test,Y_train,
                            Y_test,train['Weight'],W_test,thresh,1)
    ams_train_1, ams_test_1 = discovery_significance.ams_curve(_tree_opt_bce,X_train,X_test,Y_train,
                            Y_test,train['Weight'],W_test,thresh,1)
    
    print 'Step 6 : Graphing output saving to Graphs/'
    
    plt.figure()
    plt.grid()
    plt.stem(thresh,ams_test_0,linefmt='b-')
    plt.plot(thresh, ams_test_0,'bo',label='ROC optimizing tree')
    plt.stem(thresh,ams_test_1,linefmt='g-')
    plt.plot(thresh, ams_test_1,'go',label='BCE Minimizing tree')    
    plt.xlabel('Threshold %')
    plt.ylabel('AMS Test')
    plt.title('AMS computed on test data')
    plt.xlim(10,95)
    plt.legend(loc='best')
    plt.savefig('Graphs/AMS_Optimum.png',bbox_inches='tight')

    print 'Step 7 : Calibrating Probability Models'
    
    roc_calib_prob = tree_model.calibrate_probability(_tree_opt_roc,X_train,Y_train)
    model_prob = tree_model.model_probability(_tree_opt_roc,X_train,Y_train)
    
    plt.figure()
    plt.grid()
    plt.hist(model_prob,bins=100)
    plt.title('Frequency based probability distribution of DT')
    h = np.percentile(model_prob,80)
    plt.axvline(h,0,180,linestyle='--', color='r',label='Threshold (80th percentile)')
    plt.legend()
    plt.savefig('Graphs/Prob_model.png',bbox_inches='tight')
   
   
    plt.figure()
    plt.grid()
    plt.hist(roc_calib_prob,bins=100)
    plt.title('Calibrated probability distribution of DT')
    h = np.percentile(roc_calib_prob,80)
    plt.axvline(h,0,180,linestyle='--', color='r',label='Threshold (80th percentile)')
    plt.legend()
    plt.savefig('Graphs/Prob_calib.png',bbox_inches='tight')
  
    
    print  'Step 8 : AMS under 2 different probability models'         
                              
    ams_train_p, ams_test_p = discovery_significance.ams_curve(_tree_opt_roc,X_train,X_test,Y_train,
                            Y_test,train['Weight'],W_test,thresh,1)
    ams_train_b, ams_test_b = discovery_significance.ams_curve(_tree_opt_roc,X_train,X_test,Y_train,
                            Y_test,train['Weight'],W_test,thresh,2)
    
    plt.figure()    
    plt.grid()
    plt.stem(thresh,ams_test_p,linefmt='b-')
    plt.plot(thresh, ams_test_p,'bo',label='Default probability')
    plt.stem(thresh,ams_test_b,linefmt='r-')
    plt.plot(thresh, ams_test_b,'ro',label='Calibrated probability')    
    plt.xlabel('Threshold %')
    plt.ylabel('AMS Test')
    plt.title('AMS computed on test data')
    plt.xlim(10,95)
    plt.legend(loc='best')
    plt.savefig('Graphs/AMS_Probability_Compare.png',bbox_inches='tight')