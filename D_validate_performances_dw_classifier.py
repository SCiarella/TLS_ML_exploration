import numpy as np
import glob
import math
import gzip
import io
import os
import sys
import fnmatch
import pickle
import pandas as pd
import autogluon as ag
from autogluon.tabular import TabularDataset, TabularPredictor
import time
import multiprocess as mp
import myparams


# This code takes validates the dw classifier


if __name__ == "__main__":
    M = myparams.M
    T = myparams.T
    Tlabel = str(T).replace('.','')
    print('\n*** Requested to validate the dw classifier at T={} (M={})'.format(T,M))
    save_path='MLmodel/dw-classification-M{}-T{}'.format(M,Tlabel)
    validation_set = pd.read_feather('MLmodel/dw-classifier-validation-set-M{}-T{}.feather'.format(M,Tlabel))
    training_set = pd.read_feather('MLmodel/dw-classifier-training-set-M{}-T{}.feather'.format(M,Tlabel))
    
    # ********* RESULTS OVER THE TRAINING SET
    label= 'is_dw'
    training_set = training_set.sort_values(label,ascending=False)
    training_set_nolab = training_set.drop(columns=['i','j','conf',label])
    # * Convert to float to have optimal performances!
    training_set_nolab = TabularDataset(training_set_nolab).astype(float)
    y_true_val = training_set[label]  # values to predict
    
    
    # Load the model
    predictor = TabularPredictor.load(save_path) 
    predictor.persist_models()
    # predict
    y_pred_by_AI = predictor.predict(training_set_nolab)
    
    print('\n\nPerformances over the training set:')
    perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
    print(perf)
    
    error=0
    correct=0
    dw=0
    index=0
    for i_true, i_pred in zip(y_true_val,y_pred_by_AI):
        if i_true ==1:
            dw+=1
        if i_true!=i_pred:
            error +=1
        else:
            correct +=1
        index+=1
    print('{} are correct and {} are wrong'.format(correct, error))
    # check which class has more errors
    training_set['dw_pred'] = y_pred_by_AI
    training_set['error'] = training_set['is_dw']-training_set['dw_pred']
    error_dw = len(training_set[(training_set['is_dw']>0)&(training_set['error']!=0)])
    error_nondw = len(training_set[(training_set['is_dw']<1)&(training_set['error']!=0)])
    print('of which I mislabeled {} dw and {} non-dw'.format(error_dw, error_nondw))
    
    try:
        with open("output_ML/dw_classifier_performances.txt",'w+') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print('\t***Performances over the training set:')
            print(perf)
            print('Of the {} predictions {} are correct and {} are wrong ({} of the data are dw)'.format(len(training_set),correct, error,dw))
            print('of which I mislabeled {} dw and {} non-dw'.format(error_dw, error_nondw))
            sys.stdout = sys.__stdout__
    except Exception as error:
        print('Error: {}'.format(Exception))
        sys.exit()
    
    
    
    # ********* RESULTS OVER THE TEST SET
    # * Convert to float to have optimal performances!
    validation_set= validation_set.sort_values(label,ascending=False)
    validation_set_nolab = validation_set.drop(columns=['i','j','conf',label])
    validation_set_nolab = TabularDataset(validation_set_nolab).astype(float)
    y_true_val = validation_set[label]  # values to predict
    
    # predict
    y_pred_by_AI = predictor.predict(validation_set_nolab)
    
    print('\n\nPerformances over the test set:')
    perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
    print(perf)
    
    error=0
    correct=0
    dw=0
    index=0
    for i_true, i_pred in zip(y_true_val,y_pred_by_AI):
        if i_true ==1:
            dw+=1
        if i_true!=i_pred:
            error +=1
        else:
            correct +=1
        index+=1
    print('{} are correct and {} are wrong'.format(correct, error))
    # check which class has more errors
    validation_set['dw_pred'] = y_pred_by_AI
    validation_set['error'] = validation_set['is_dw']-validation_set['dw_pred']
    error_dw = len(validation_set[(validation_set['is_dw']>0)&   (validation_set['error']!=0)])
    error_nondw = len(validation_set[(validation_set['is_dw']<1)&(validation_set['error']!=0)])
    print('of which I mislabeled {} dw and {} non-dw'.format(error_dw, error_nondw))
    
    try:
        with open("output_ML/dw_classifier_performances.txt",'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print('\n\n\t***Performances over the test set:')
            print(perf)
            print('Of the {} predictions {} are correct and {} are wrong ({} of the data are dw)'.format(len(validation_set),correct, error,dw))
            print('of which I mislabeled {} dw and {} non-dw'.format(error_dw, error_nondw))
            sys.stdout = sys.__stdout__
    except Exception as error:
        print('Error: {}'.format(Exception))
        sys.exit()
    
