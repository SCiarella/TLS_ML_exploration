import numpy as np
import time
import glob
import pandas as pd
import csv
import gzip
import io
import os
import sys
import fnmatch
import pickle
import autogluon as ag
from autogluon.tabular import TabularDataset, TabularPredictor
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib
import myparams

# This code applied the predictor to all the pairs in class-1, and ranks them accordingly 

In_label = myparams.In_file.split('/')[-1].split('.')[0]
print('\n*** Requested to apply the predictor to {}'.format(In_label))
model_path='MLmodel/prediction-{}'.format(In_label)

# check if the model is there
if not os.path.isdir(model_path):
    print('Error: I am looking for the ML model in {}, but I can not find it. You probably have to run step3.py before this'.format(model_path))
    sys.exit()

# Load the model
predictor = TabularPredictor.load(model_path) 
predictor.persist_models()


# Read the data
outdir = 'output_ML/{}'.format(In_label)
pairs_df = pd.read_feather('output_ML/{}/classified_{}.feather'.format(In_label,In_label))
nconfs = len(pairs_df['conf'].drop_duplicates())
print('\n*** Found {} macro-configurations and a total of {} pairs\n\nStarting predictions'.format(nconfs, len(pairs_df)))

# predict the qs
X = pairs_df.drop(columns=['i','j','conf','class'])
X = TabularDataset(X)
y_pred_by_AI = predictor.predict(pairs_df)
y_pred_by_AI = np.power(10, -y_pred_by_AI)
print('The target has been predicted. Now storing results')

# store the predictions
pairs_df['target_feature']=y_pred_by_AI
pairs_df = pairs_df.sort_values(by='target_feature')
pairs_df[['conf','i','j','target_feature']].to_csv('{}/predicted_{}_allpairs.csv'.format(outdir,In_label),index=False)
all_qs_df = pairs_df.copy()


# then I exclude the pairs that are bad (according to the exact calculation)
calculation_dir='./exact_calculations/{}'.format(In_label)
if not os.path.isfile('{}/{}'.format(calculation_dir,myparams.calculations_classifier)):
    print('\n*(!)* Notice that there are no classification data\n')
else:
    class_0_pairs = pd.read_csv('{}/{}'.format(calculation_dir,myparams.calculations_classifier), index_col=0)
    class_1_pairs = pd.read_csv('{}/{}'.format(calculation_dir,myparams.calculations_predictor), index_col=0)[['conf','i','j']]

    temp_df = all_qs_df.reset_index(drop=True)
    temp_df['index'] = temp_df.index
    remove_df = temp_df.merge(class_0_pairs, how = 'inner' ,indicator=False)
    remove_df = remove_df.set_index('index')
    all_qs_df= all_qs_df.drop(remove_df.index).reset_index()
    print('\n*We know that {} of the new pairs are class-0 (from calculations), so we do not need to predict them.\nWe then end up with {} new pairs'.format(len(remove_df),len(all_qs_df)))

# then exclude the pairs for which I already calculated the exact target property
if not os.path.isfile('{}/{}'.format(calculation_dir,myparams.calculations_predictor)):
    print('\n*(!)* Notice that there are no prediction data\n')
else:
    calculated_pairs = pd.read_csv('{}/{}'.format(calculation_dir,myparams.calculations_predictor), index_col=0)

    temp_df = all_qs_df.reset_index(drop=True)
    temp_df['index'] = temp_df.index
    remove_df = temp_df.merge(calculated_pairs, how = 'inner' ,indicator=False)
    remove_df = remove_df.set_index('index')
    all_qs_df= all_qs_df.drop(remove_df.index)
    all_qs_df= all_qs_df.reset_index(drop=True)
    print('\n*For {} of the new pairs we already run the calculations, so we do not need to predict them.\nWe then finish with {} new pairs'.format(len(remove_df),len(all_qs_df)))

# Storing
all_qs_df[['conf','i','j','target_feature']].to_csv('{}/predicted_{}_newpairs.csv'.format(outdir,In_label),index=False)
