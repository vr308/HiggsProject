# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:22:07 2016

@author: vr308
"""

import numpy as np
import sys
import matplotlib.pylab as plt
import timeit

import preprocessing
import sampling
import cross_validation
import discovery_significance
import visualization

#os.chdir(os.getcwd()+ '/higgsSVM')

train_sample_type = sys.argv[1]
mode = sys.argv[2]
code_path = sys.argv[3]

def timer(start_time):

    return np.round(timeit.default_timer() - start_timer_preprocess,2)


def print_guidelines(mode):

    if (mode == 'train'):

        print 'NOTE : Full training, cross validation and optimization will be conducted before testing'

    if (mode == 'test'):

        print 'NOTE : Only testing will be conducted in this mode'

if __name__ == "__main__":

    print_guidelines(mode)

#---------------------------------------------------------------------------
# PREPROCESSING :
#---------------------------------------------------------------------------

    print '\n' + '-------------------------Starting PREPROCESSING-------------------------------' + '\n'

    start_timer_preprocess = timeit.default_timer()

    print 'Step 1 : Loading data'
    df = preprocessing.load_data(path=code_path +'/Data/')

    print 'Step 2 : Dropping redundant features'
    df = preprocessing.drop_features(df)

    print 'Step 3 : Cleaning up missing value tags : -999.0'
    df = preprocessing.drop_missing_values(df)

    train, test = preprocessing.train_test_split(df,perc=0.80)

    elapsed = timer(start_timer_preprocess)

    print'\n' + '---------------------Finished PREPROCESSING stage, took ' + str(elapsed) + ' seconds----------' + '\n'

#----------------------------------------------------------------------------
# ----------------------SAMPLING : Training with 2 types of samples---------
# 1) Uniform sample
# 2) Choice sample
#----------------------------------------------------------------------------

    print '--------------------------Starting SAMPLING-----------------------------------' + '\n'


    start_timer_sampling = timeit.default_timer()

    print 'Step 1 : Scaling features, drawing uniform sample'
    train_uniform = sampling.get_training_sample(train,sample_type='uniform',normalize=True)

    print 'Step 2 : Scaling features, drawing choice sample (+/- 1.6*sd) around mean for each feature '
    train_choice = sampling.get_training_sample(train,sample_type='choice',normalize=True)

    print 'Step 3 : Preparing training and test data'

    if (train_sample_type == 'choice_sample'):
            train_sample = train_choice
    else:
            train_sample = train_uniform

    train_sample = preprocessing.generate_balanced_weights(train_sample)

    X_train = preprocessing.get_features(train_sample)
    Y_train = train_sample['Label']
    W_train = train_sample['Weight']
    W_train_balanced = X_train.pop('W_balanced')

    test = preprocessing.generate_balanced_weights(test)

    X_test = preprocessing.normalize_fit_test(preprocessing.normalize(train)[1],test.drop(labels=['W_balanced'],axis=1))
    X_test = preprocessing.get_features(X_test)
    Y_test = test['Label']
    W_test = test['Weight']
    W_test_balanced = test.pop('W_balanced')

    elapsed = timer(start_timer_sampling)

    print '\n' + '------------------------Finished SAMPLING stage, took ' +  str(elapsed) + ' seconds------------' + '\n'

#----------------------------------------------------------------------------
#----------------------TRAINING---------------------------------------------
# Base 10 grid search for C / gamma
# Base 2 grid search for C / gamma
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

    print '-------------------------Starting TRAINING and CROSS VALIDATION----------------------------------------' + '\n'

    if (mode == 'train'):

        start_timer_training = timeit.default_timer()

        print 'Step 1 : Optimizing parameters C and gamma (RBF Kernel) on a base10 & base2 grid for ROC and balanced classification error ' + '\n'

        # Base 2 grid optimization is ommited here due to compute time, in order to try it out, despite the compute time, extend grid_density list to [2,10].

        grid_density = [10]

        for i in grid_density:

            C_range, gamma_range = cross_validation.get_mesh_grid(i)

            print ' Step 1 : Grid search for ROC optimal and R(f) optimal parameters'
            grid_scores=cross_validation.grid_search_metric(X_train,Y_train,'roc_auc',
                                                            C_range,gamma_range, train_sample_type,i)
            grid_error = cross_validation.grid_search_error(X_train,Y_train,W_train_balanced,C_range,gamma_range, train_sample_type,i)

            print 'Step 2 : Saving visualizations'
            visualization.plot_grid_scores(grid_scores, C_range,gamma_range,
                                         'ROC_AUC',train_sample_type,plt.cm.hot,vmin=0.8,vmax=0.95,grid_density=i)
            visualization.plot_grid_scores(grid_error,C_range,gamma_range,'Balanced_Classification_Error',train_sample_type,plt.cm.jet,vmin=0.0016,vmax=0.004,grid_density=i)

        elapsed = timer(start_timer_training)

    print '\n' + '----------Finished TRAINING and CROSS VALIDATION, took ' +  str(elapsed/60) + ' minutes--------------' + '\n'

#----------------------------------------------------------------------------
#----------------------TESTING-----------------------------------------------
#----------------------------------------------------------------------------

    print '--------------------------------Starting TESTING------------------------------------------' + '\n'

    start_timer_testing = timeit.default_timer()

    #tuned_C = [10,100,1000]
    #tuned_gamma = [0.01,0.0004,0.00007]

    #tuned_C = [12]
    #tuned_gamma = [0.008,0.01,0.012]

    X_train_sub = X_train.drop(labels=['A'],axis=1)
    X_train_plus = X_train

    X_test_sub = X_test.drop(labels=['A'],axis=1)
    X_test_plus = X_test

    modelA = cross_validation.fit_svm(X_train_plus,Y_train,'rbf', C=12, gamma=0.01)
    modelNA = cross_validation.fit_svm(X_train_sub,Y_train,'rbf', C=12, gamma=0.01)
    thresholds,ams_train_A,ams_test_A=discovery_significance.ams_curve(modelA,X_train_plus,X_test_plus,Y_train,Y_test,W_train,W_test)
    thresholds,ams_train_NA,ams_test_NA=discovery_significance.ams_curve(modelNA,X_train_sub,X_test_sub,Y_train,Y_test,W_train,W_test)


#    plt.figure()
#    plt.grid()
#    plt.stem(thresholds, ams[0],linefmt='k-')
#    plt.stem(thresholds, ams[1],markerfmt='ro',linefmt='k-')
#    plt.stem(thresholds, ams[2],markerfmt='go',linefmt='k-')
#    plt.legend(('0.008','0.01','0.012'),title='Gamma',loc=2)
#    plt.title('AMS scores by thresholds for 3 different gamma values')
#    plt.xlim(75,92)
#    plt.ylim(0,5)
#    plt.savefig('AMS_by_gamma_A',format='png')
#
    #visualization.plot_ams_curve(thresh,ams_test0,'Test',train_sample_type)

    print('Peak AMS -- ' + str(max(ams_test0)))

    elapsed = timer(start_timer_testing)

    print '\n' + '------------------------Finished TESTING, took ' +  str(elapsed) + ' seconds----------------------' + '\n'












#
#    model0 = cross_validation.fit_svm(X_sub,Y_train,'rbf', C=tuned_C[1], gamma=tuned_gamma[1])
#    thresh, ams_train, ams_test0=discovery_significance.ams_curve(model0,X_sub,X_test_sub,Y_train,Y_test,W_train,W_test)
#
#    plt.figure()
#    plt.grid()
#    plt.stem(thresh,ams_test0,linefmt='b-',markerfmt='bo')
#    plt.stem(thresh,ams_test1,linefmt='g-',markerfmt='go')
#    plt.xlabel('Threshold %')
#    plt.ylabel('AMS Test')
#    plt.title('AMS curves computed on test data with / without feature A')
#    plt.xlim(80,90)



#
#    plt.figure()
#    plt.grid()
#    plt.stem(thresh,ams_test1,linefmt='b-',markerfmt='bo')
#    plt.stem(thresh,ams_test2,linefmt='g-',markerfmt='go')
#    plt.stem(thresh,ams_test3[2],linefmt='r-',markerfmt='ro')
#    plt.xlabel('Threshold %')
#    plt.ylabel('AMS Test')
#    plt.title('AMS curves computed on test data for 3 sets of parameters')
#    plt.xlim(80,90)