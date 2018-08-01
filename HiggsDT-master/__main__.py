# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:23:02 2016

@author: vr308

"""

import numpy as np
import timeit
import matplotlib.pylab as plt
import sys

import preprocessing
import cross_validation

mode = sys.argv[2]
code_path = sys.argv[3]

def timer(start_time):
    
    return np.round(timeit.default_timer() - start_timer_preprocess,2)
    
if __name__ == "__main__":

#---------------------------------------------------------------------------
# PREPROCESSING : 
#---------------------------------------------------------------------------

    print '\n' + '-------------------------Starting PREPROCESSING-------------------------------' + '\n'
    
    start_timer_preprocess = timeit.default_timer() 
        
    print 'Step 1 : Loading the data '
    df = preprocessing.load_data(code_path + '/Data/')

    print 'Step 2 : Dropping redundant features'    
    df = preprocessing.drop_features(df)

    print 'Step 3 : Cleaning up missing value tags : -999.0'    
    df = preprocessing.drop_missing_values(df)
    
    train, test = preprocessing.train_test_split(df,perc=0.75)
    
    print 'Step 4 : Preparing Weights for training by re-balancing '
    
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
    
    elapsed = timer(start_timer_preprocess)
    
    print'\n' + '---------------------Finished PREPROCESSING stage, took ' + str(elapsed) + ' seconds----------' + '\n'
    
#---------------------------------------------------------------------------
# TRAINING 
#---------------------------------------------------------------------------
    
    print '\n' + '-------------------------Starting TRAINING-------------------------------' + '\n'

    start_timer_training = timeit.default_timer() 
    
    print 'Step 1 : Performing a grid search on 5 tree parameters for optimal parameters'
        
    max_features_range = [5,10,None]
    max_depth_range = [None] + [5,10,15,50,70]          
    min_samples_leaf_range = [1,50,150,200,500] 
    min_samples_split_range = [10,40,100,200,500]
    criterion_range = ['gini','entropy']
    
    metric = 'roc_auc' 
        
    grid_metric = cross_validation.grid_search_metrics(X_train,Y_train,'roc_auc',criterion_range,max_depth_range,
                            min_samples_leaf_range,min_samples_split_range, max_features_range)     
    
    #grid_error = cross_validation.grid_search_error(X_train,Y_train,W_train,criterion_range,
                                #max_depth_range,min_samples_leaf_range,min_samples_split_range,max_features_range)    
    
       
    print 'Step 2: Saving visualizations'
    
    def plot_grid_params(grid_score_df,metric,split_metric,legend_loc):
        
        splits = list(np.unique(grid_score_df[split_metric]))
        plt.figure(figsize=(6,6))
        plt.grid()      
        plt.xlabel(metric)
        plt.ylabel('count')
        plt.title('Distribution of ' + metric + ' by ' + split_metric)
        colors = ['b','g','r','m','y','c']
        for i,j in zip(splits,colors):
            group = grid_score_df[grid_score_df[split_metric] == i]['mean_score']
            plt.hist(group,histtype='stepfilled',alpha=1,label=str(i),normed=True,color=j)
            plt.axvline(x=np.mean(group),linestyle='--',color=j)
        plt.legend(title=split_metric,loc=legend_loc)
        plt.savefig('Graphs/' + metric + '_' + split_metric + '.png',bbox_inches='tight')
        

                
#        split_metric = 'max_features'
#        metric = 'balanced classification error'
#        plot_grid_params(grid_error,metric,split_metric,1)
    
              
    elapsed = timer(start_timer_training)

    print '\n' + '----------Finished TRAINING and CROSS VALIDATION, took ' +  str(elapsed/60) + ' minutes--------------' + '\n'





