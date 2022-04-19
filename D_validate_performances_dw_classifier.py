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
from sklearn.model_selection import train_test_split
import time
import multiprocessing as mp


# This code takes all the available data (results from the NEB) and if they are more than the data that we already used to train the model, we retrain it 


# We need the desired M for the target df
try:
    with open("M_val.txt") as f:
        M=int(f.readlines()[0].strip('\n'))
        print('M={}'.format(M))
except Exception as error:
    print('Error: {}'.format(Exception))
    sys.exit()

save_path='MLmodel/dw-classification-M{}'.format(M)
validation_set = pd.read_pickle('MLmodel/dw-classifier-validation-set-M{}.pickle'.format(M))
training_set = pd.read_pickle('MLmodel/dw-classifier-training-set-M{}.pickle'.format(M))

# ********* RESULTS OVER THE TRAINING SET
label= 'is_dw'
# * Convert to float to have optimal performances!
training_set = TabularDataset(training_set).astype(float)
y_true_val = training_set[label]  # values to predict
y_true_val = y_true_val.sort_values(ascending=False)
training_set= training_set.sort_values(label,ascending=False)
training_set_nolab = training_set.drop(columns=[label])  # delete label column to prove we're not cheating

# Load the model
predictor = TabularPredictor.load(save_path, verbosity=3) 
predictor.persist_models()
# predict
y_pred_by_AI = predictor.predict(training_set_nolab)

print('\n\nPerformances over the training set:')
perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
print(perf)

error=0
correct=0
dw=0
for i_true, i_pred in zip(y_true_val,y_pred_by_AI):
    if i_true ==1:
        dw+=1
    if i_true!=i_pred:
        error +=1
    else:
        correct +=1
print('{} are correct and {} are wrong'.format(correct, error))

try:
    with open("output_ML/dw_classifier_performances.txt",'w+') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('\t***Performances over the training set:')
        print(perf)
        print('Of the {} predictions {} are correct and {} are wrong ({} were dw)'.format(len(training_set),correct, error,dw))
        sys.stdout = sys.__stdout__
except Exception as error:
    print('Error: {}'.format(Exception))
    sys.exit()



# ********* RESULTS OVER THE TEST SET
# * Convert to float to have optimal performances!
validation_set=validation_set.drop(columns=['i','j','conf'])
validation_set = TabularDataset(validation_set).astype(float)
y_true_val = validation_set[label]  # values to predict
y_true_val = y_true_val.sort_values(ascending=False)
validation_set= validation_set.sort_values(label,ascending=False)
validation_set_nolab = validation_set.drop(columns=[label])  # delete label column to prove we're not cheating

# predict
y_pred_by_AI = predictor.predict(validation_set_nolab)

print('\n\nPerformances over the test set:')
perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
print(perf)

error=0
correct=0
dw=0
for i_true, i_pred in zip(y_true_val,y_pred_by_AI):
    if i_true ==1:
        dw+=1
    if i_true!=i_pred:
        error +=1
    else:
        correct +=1
print('{} are correct and {} are wrong'.format(correct, error))

try:
    with open("output_ML/dw_classifier_performances.txt",'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('\n\n\t***Performances over the test set:')
        print(perf)
        print('Of the {} predictions {} are correct and {} are wrong ({} were dw)'.format(len(validation_set),correct, error,dw))
        sys.stdout = sys.__stdout__
except Exception as error:
    print('Error: {}'.format(Exception))
    sys.exit()

