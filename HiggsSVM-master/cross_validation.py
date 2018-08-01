# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:23:02 2016

@author: vr308
"""


import numpy as np
import sklearn.svm as svm
import itertools
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import discovery_significance


def fit_svm(features, labels,kernel, C, gamma):
        model = svm.SVC(kernel = kernel, C=C, gamma=gamma, probability=True)
        model.fit(features, labels)
        return model
    

def balanced_classification_error(model,X,Y,W):
    
        Y_pred = model.predict(X)
        Y_miss = (Y != Y_pred)
        return sum(W*Y_miss)


def grid_search_metric(X_train,Y_train,metric,C_range, gamma_range,train_sample_type,grid_density):
    
    param_grid = dict(gamma=gamma_range, C=C_range)
    param_combinations =  list(itertools.product(C_range,gamma_range))
    cv = StratifiedShuffleSplit(Y_train, n_iter=3, test_size=0.25, random_state=87)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv,scoring=metric,verbose=True,n_jobs=1)
    grid.fit(X_train, Y_train)
    
    # grid_scores_ contains parameter settings and scores
    # We extract just the scores
    
    scores = [x[1] for x in grid.grid_scores_]
    grid_scores = np.array(np.round(scores,4)).reshape(len(C_range), len(gamma_range))
    string_scores = [str(i) for i in grid.grid_scores_]
    
    filename = 'Results/grid_scores_base' + str(grid_density) + '_' + metric + '_' + train_sample_type + '_sample.txt'
    print ('Writing output to file : ' + filename + '\n')
    
    target = open(filename,'w')
    target.write("3-fold cross-validation grid search for C and gamma for metric : " + metric+'\n')
    target.write("Training sample type : " + train_sample_type + '\n')
    target.writelines(['%s\n' % item for item in string_scores])

    param_df = pd.DataFrame({'scores' : scores, 'params' : param_combinations})    
    best_params = param_df[param_df.scores > 0.85]
    
    target.write('\n' + 'Best parameters with score > 0.85')
    target.write(str(best_params))
    target.close()

    return grid_scores#, param_combinations, best_params

def get_mesh_grid(base):
    
    C_range = np.round(np.logspace(-1,3,5),2)                  
    if (base == 10):
        gamma_range = np.round(np.logspace(-4,0,5),6)
    if(base == 2):
        gamma_range = np.round(np.logspace(-14,-1,15,base=2),6) 
    return C_range, gamma_range

def get_classifiers(X_train, Y_train ,best_params):
    
    classifiers = []
    j = 0
    print ('Saving classifiers for best parameters (C,gamma) yielded in the base2 grid search')
    for i in best_params:
             j = j+1   
             (C ,gamma) = i
             print ('Iteration %s of %s  with params C = %0.2f and gamma =  %0.6f' % (j,len(best_params),C,gamma))
             model = fit_svm(X_train, Y_train,kernel='rbf',C=C, gamma=gamma)
             classifiers.append(model)
    C, gamma = zip(*np.asarray(best_params))
    return classifiers, np.unique(C), np.unique(gamma)


def grid_ams(classifiers,X_train,Y_train,X_test, Y_test, W_train,W_test,train_sample_type,C_range, gamma_range):

    filename = 'Results/grid_ams_' + train_sample_type + '.txt'
    print ('Writing output to file : ' + filename)
    
    target = open(filename,'w')
    ams_train_scores = []
    ams_test_scores = []
    j = 0 
    for clf in classifiers:
            j = j+1
            target.write ('Iteration %s of %s with params C = %0.2f and gamma =  %0.4f' % (j,len(classifiers),clf.C,clf.gamma) + '\n')
            ams_train, ams_test =  discovery_significance.ams_score(clf,X_train,X_test,Y_train,Y_test,W_train,W_test,84)
            target.writelines('AMS Train : ' + str(ams_train) + ' ') 
            target.writelines('AMS Test : ' + str(ams_test) + ' ' + '\n')            
            ams_train_scores.append(ams_train)
            ams_test_scores.append(ams_test) 
    best_classifier = classifiers[ams_test_scores.index(max(ams_test_scores))]
    target.close()
    
    #ams_train_grid = np.array(ams_train_scores).reshape(len(C_range),len(gamma_range))
    #ams_test_grid = np.array(ams_test_scores).reshape(len(C_range),len(gamma_range))
   
    return ams_train_scores, ams_test_scores, best_classifier
    

def grid_search_error(X_train,Y_train,W_balanced,C_range,gamma_range,train_sample_type,grid_density):
    
    param_list=list(itertools.product(C_range,gamma_range))
    
    ksplit = StratifiedKFold(Y_train, n_folds=2)
    
    mean_validation_error = []
    
    mean_error_values = []
    std_error_values = []
    param_strings = []
    
    count = 0
    total = len(param_list)
    
    for p in param_list: 
        
        count += 1
        print ('Iteration ' + str(count) + ' of '  + str(total) + '\n')
        C = p[0]
        gamma = p[1]
       
        param_string = "params: {'C' : " + str(C) + ", 'gamma' : "  + str(gamma) + "}"
        
        iteration_errors = []
        
        for learn_index, validate_index in ksplit :
            
            X_learn , X_validate = np.asarray(X_train)[learn_index], np.asarray(X_train)[validate_index]
            Y_learn , Y_validate = Y_train[learn_index], Y_train[validate_index]
            W_learn , W_validate = W_balanced[learn_index], W_balanced[validate_index]
            
            svm_model = fit_svm(X_learn, Y_learn,'rbf',C,gamma)
                                        
            iteration_errors.append(balanced_classification_error(svm_model, X_validate, Y_validate, W_validate))
        
        mean_error = np.round(np.mean(iteration_errors),5)
        std_error = np.round(np.std(iteration_errors),5)
        
        mean_error_values.append(mean_error)
        std_error_values.append(std_error)  
        param_strings.append(param_string)
        
        mean_validation_error.append(['mean: ' + str(mean_error) + ', std: ' + str(std_error) + ', ' + param_string])      
    
    string_scores = [str(i) for i in mean_validation_error]
    filename = 'Results/grid_scores_base' + str(grid_density) + '_' +  'balanced_classification_error' + train_sample_type + '.txt'
    print ('Writing output to file : ' + filename + '\n')
    
    target = open(filename,'w')
    target.write("3-fold cross-validation grid search for C and gamma " + '\n')
    target.writelines(['%s\n' % item for item in string_scores])    
    target.close()
    
    grid_error = np.array(np.round(mean_error_values,4)).reshape(len(C_range), len(gamma_range))
   
    return grid_error
    
    
