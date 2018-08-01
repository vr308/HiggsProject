# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:23:02 2016

@author: vr308
"""

from sklearn.tree import tree
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import discovery_significance
from operator import itemgetter
import pandas as pd

import tree_model
import numpy as np
import itertools

    
def grid_search_metrics(X_train,Y_train,metric,criterion_range, max_depth_range, min_samples_leaf_range,min_samples_split_range, max_features_range):
         
    param_grid = dict(max_depth=max_depth_range, 
                      min_samples_leaf=min_samples_leaf_range,
                      min_samples_split= min_samples_split_range,
                      criterion=criterion_range, 
                      max_features=max_features_range)
 
    cv = StratifiedShuffleSplit(Y_train, n_iter=3, test_size=0.3)
    grid = GridSearchCV(tree.DecisionTreeClassifier(criterion=criterion_range,splitter='best',class_weight='auto'), 
                        param_grid=param_grid, cv=cv,scoring=metric,verbose=True,n_jobs=1)
    grid.fit(X_train, Y_train)
    
    # grid_scores_ contains parameter settings and scores
    # We extract just the scores
    #scores = [x[1] for x in grid.grid_scores_]
    string_scores = [str(i) for i in grid.grid_scores_]
    print("The best parameters for " + metric + " are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
          
    
    filename = 'Results/grid_scores_' + metric + '_.txt'
    print ('Writing output to file : ' + filename + '\n')
    
    target = open(filename,'w')
    target.write("3-fold cross-validation grid search for max_depth, min_samples_leaf, criterion, min_samples_split, max_features for metric : " + metric+'\n')
    target.writelines(['%s\n' % item for item in string_scores])    
    target.close()
    
    grid_summary_report_metric(grid.grid_scores_, metric, top=3)

    return grid_score_dataframe(grid)

def grid_score_dataframe(grid_metric):
    
    mean_scores = []
    max_features = []
    min_samples_split = []
    max_depth = []
    min_samples_leaf = []
    criterion = []
    for i in np.arange(0,len(grid_metric.grid_scores_)):
        mean_scores.append(grid_metric.grid_scores_[i][1])    
        max_features.append(grid_metric.grid_scores_[i][0]['max_features'])
        max_depth.append(grid_metric.grid_scores_[i][0]['max_depth'])
        min_samples_split.append(grid_metric.grid_scores_[i][0]['min_samples_split'])
        min_samples_leaf.append(grid_metric.grid_scores_[i][0]['min_samples_leaf'])
        criterion.append(grid_metric.grid_scores_[i][0]['criterion'])
    grid_metric_df = pd.DataFrame(np.c_[mean_scores, max_features, max_depth, min_samples_split,
                                   min_samples_leaf,criterion], columns=['mean_score','max_features','max_depth','min_samples_split','min_samples_leaf','criterion'])
    grid_metric_df = grid_metric_df.fillna(value=0)
    return grid_metric_df               
                                   
def grid_search_error(X_train,Y_train,W_train,criterion_range, max_depth_range, min_samples_leaf_range,min_samples_split_range, max_features_range):
    
    param_list=list(itertools.product(max_features_range,
                                      min_samples_split_range,
                                      criterion_range,
                                      max_depth_range,
                                      min_samples_leaf_range))
    
    ksplit = StratifiedKFold(Y_train, n_folds=3)
    
    mean_validation_error = []
    
    mean_error_values = []
    std_error_values = []
    param_strings = []
    
    count = 0
    total = len(param_list)
    for p in param_list: 
        count += 1
        print ('Iteration ' + str(count) + ' of '  + str(total) + '\n')
        max_features = p[0]
        min_samples_split = p[1]
        criterion = p[2]
        max_depth = p[3]
        min_samples_leaf = p[4]
       
        param_string = "params: {'max_features' : " + str(p[0]) + ", 'min_samples split' : "  + str(p[1]) + ", 'criterion' : " + str(p[2])  + ", 'max_depth' : " + str(p[3]) + ", 'min_samples_leaf' : " + str(p[4]) + "}"
        
        iteration_errors = []
        
        for learn_index, validate_index in ksplit :
            X_learn , X_validate = np.asarray(X_train)[learn_index], np.asarray(X_train)[validate_index]
            Y_learn , Y_validate = Y_train[learn_index], Y_train[validate_index]
            W_learn , W_validate = W_train[learn_index], W_train[validate_index]
            
            _tree = tree_model.fit_tree(X_learn, Y_learn,W_learn,
                                        criterion,
                                        max_depth,
                                        min_samples_leaf,
                                        min_samples_split,
                                        max_features)
                                        
            iteration_errors.append(tree_model.balanced_classification_error(_tree, X_validate, Y_validate, W_validate))
        mean_error = np.round(np.mean(iteration_errors),5)
        std_error = np.round(np.std(iteration_errors),5)
        
        mean_error_values.append(mean_error)
        std_error_values.append(std_error)  
        param_strings.append(param_string)
        
        mean_validation_error.append(['mean: ' + str(mean_error) + ', std: ' + str(std_error) + ', ' + param_string])      
    
    string_scores = [str(i) for i in mean_validation_error]
    filename = 'Results/grid_scores_balanced_classification_error.txt'
    print ('Writing output to file : ' + filename + '\n')
    
    target = open(filename,'w')
    target.write("3-fold cross-validation grid search for max_depth, min_samples_leaf, criterion, min_samples_split, max_features for metric : balanced classification error" + '\n')
    target.writelines(['%s\n' % item for item in string_scores])    
    target.close()
    
    max_features, min_samples_split, criterion, max_depth, min_samples_leaf = zip(*param_list)
    grid_error = pd.DataFrame(np.c_[mean_error_values,
                                    std_error_values,
                                   max_features,
                                   min_samples_split,
                                   criterion,
                                   max_depth,
                                   min_samples_leaf,
                                   param_strings],columns=['mean_score','std','max_features','min_samples_split','criterion','max_depth','min_samples_leaf','params'])
    
    grid_summary_report_error(grid_error.sort(column=['mean_score']),'balanced classification error', top=3)
    grid_error = grid_error.fillna(value=0)
    return grid_error
    
def grid_search_ams(X_train,Y_train,W_train,criterion_range, max_depth_range, min_samples_leaf_range,min_samples_split_range, max_features_range,cuts,prob,bm):
    
    param_list=list(itertools.product(max_features_range,
                                      min_samples_split_range,
                                      criterion_range,
                                      max_depth_range,
                                      min_samples_leaf_range))
    
    param_list = param_list[0:3]
    ksplit = StratifiedKFold(Y_train, n_folds=3)
    
    mean_validation_error = []
    
    mean_ams_values = []
    std_ams_values = []
    param_strings = []    
    
    count = 0
    total = len(param_list)
    for p in param_list: 
        count += 1
        print ('Iteration ' + str(count) + ' of '  + str(total) + '\n')
        max_features = p[0]
        min_samples_split = p[1]
        criterion = p[2]
        max_depth = p[3]
        min_samples_leaf = p[4]
       
        param_string = "params: {'max_features' : " + str(p[0]) + ", 'min_samples split' : "  + str(p[1]) + ", 'criterion' : " + str(p[2])  + ", 'max_depth' : " + str(p[3]) + ", 'min_samples_leaf' : " + str(p[4]) + "}"
        
        ams = []
        
        for learn_index, validate_index in ksplit :
            X_learn , X_validate = np.asarray(X_train)[learn_index], np.asarray(X_train)[validate_index]
            Y_learn , Y_validate = Y_train[learn_index], Y_train[validate_index]
            W_learn , W_validate = W_train[learn_index], W_train[validate_index]
            
            _tree = tree_model.fit_tree(X_learn, Y_learn,W_learn,
                                        criterion,
                                        max_depth,
                                        min_samples_leaf,
                                        min_samples_split,
                                        max_features)
                                        
            ams.append(discovery_significance.average_ams(_tree,X_train,X_validate, Y_train,Y_validate, W_train,W_validate,cuts,prob,bm)[1])
        mean_error = np.round(np.mean(ams),5)
        std_error = np.round(np.std(ams),5)
        
        mean_ams_values.append(mean_error)
        std_ams_values.append(std_error)
        param_strings.append(param_string)        
        
        mean_validation_error.append(['mean: ' + str(mean_error) + ', std: ' + str(std_error) + ', ' + param_string])      
    
    string_scores = [str(i) for i in mean_validation_error]
    filename = 'Results/grid_scores_ams.txt'
    print ('Writing output to file : ' + filename + '\n')
    
    target = open(filename,'w')
    target.write("3-fold cross-validation grid search for max_depth, min_samples_leaf, criterion, min_samples_split, max_features for metric : ams" + '\n')
    target.writelines(['%s\n' % item for item in string_scores])    
    target.close()
    
    #grid_ams = pd.DataFrame(np.c_[mean_error_values,std_error_values,param_strings],columns=['mean_score','std','params'])
    max_features, min_samples_split, criterion, max_depth, min_samples_leaf = zip(*param_list)
    grid_ams = pd.DataFrame(np.c_[mean_ams_values,
                                  std_ams_values,
                                   max_features,
                                   min_samples_split,
                                   criterion,
                                   max_depth,
                                   min_samples_leaf,
                                   param_strings],columns=['mean_score','std','max_features','min_samples_split','criterion','max_depth','min_samples_leaf','params'])
    grid_summary_report_error(grid_ams.sort(column=['mean_score'],ascending=False),'ams', top=3)
    return grid_ams


def grid_summary_report_metric(grid_scores, metric, top=3):
    
    filename = 'Results/top_params_' + metric + '_.txt'
    print ('Writing output to file : ' + filename + '\n')
    
    target = open(filename,'w')
    target.write("Top parameters report for metric : " + metric+'\n') 
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:top]
    for i, score in enumerate(top_scores):
        target.writelines("Model with rank: {0}".format(i + 1) + ' ')
        target.writelines(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        target.writelines("Parameters: {0}".format(score.parameters))
        target.write('\n')
    
    return top_scores[0].parameters
    
def grid_summary_report_error(grid_scores, metric, top=3):
    
    filename = 'Results/top_params_' + metric + '_.txt'
    print ('Writing output to file : ' + filename + '\n')
    target = open(filename,'w')
    target.write("Top parameters report for metric : " + metric+'\n') 
    top_scores = grid_scores[:top]
    for i in np.arange(0,3):
        target.writelines("Model with rank: {0}".format(i + 1) + ' ')
        target.write('mean: ' + str(top_scores.irow(i)['mean_score']) + ' ')
        target.write('std: ' + str(top_scores.irow(i)['std']) + ' ')
        target.write('params: ' + str(top_scores.irow(i)['params']) + '\n')
        target.write('\n')
    
    return top_scores    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#def grid_ams(classifiers,X_train,Y_train,X_test, Y_test, W_train,W_test):
#
#    ams_train_scores = []
#    ams_test_scores = []
#    j = 0 
#    for clf in classifiers:
#            j = j+1
#            print ('Iteration %s of %s  with params max_depth = %s and min_samples_leaf =  %s' 
#             % (j,len(param_grid),max_depth,min_samples_leaf))
#            ams_train, ams_test =  discovery_significance.ams_score(clf,X_train,X_test,Y_train,Y_test,
#                                                                    W_train,W_test,85)
#            ams_train_scores.append(ams_train)
#            ams_test_scores.append(ams_test)
#    return ams_train, ams_test
#
#def grid_ams(classifiers,X_train,Y_train,X_test, Y_test, W_train,W_test):
#
#    filename = 'Results/grid_ams.txt'
#    print ('Writing output to file : ' + filename)
#    
#    target = open(filename,'w')
#    ams_train_scores = []
#    ams_test_scores = []
#    j = 0 
#    for clf in classifiers:
#            j = j+1
#            target.write ('Iteration %s of %s  with params max_depth = %s and min_samples_leaf =  %s' 
#             % (j,len(param_grid),max_depth,min_samples_leaf)+ '\n')            
#            ams_train, ams_test =  discovery_significance.ams_score(clf,X_train,X_test,Y_train,Y_test,W_train,W_test,84)
#            target.writelines('AMS Train : ' + str(ams_train) + ' ') 
#            target.writelines('AMS Test : ' + str(ams_test) + ' ' + '\n')            
#            ams_train_scores.append(ams_train)
#            ams_test_scores.append(ams_test) 
#    best_classifier = classifiers[ams_test_scores.index(max(ams_test_scores))]
#    target.close()
    
    #ams_train_grid = np.array(ams_train_scores).reshape(len(C_range),len(gamma_range))
    #ams_test_grid = np.array(ams_test_scores).reshape(len(C_range),len(gamma_range))
   
#    return ams_train_scores, ams_test_scores, best_classifier

