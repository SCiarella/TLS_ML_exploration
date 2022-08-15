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

# This code runs the qs prediction for all the pairs that have been predicted to be dw 


M = myparams.M
T = myparams.T
Tlabel = str(T).replace('.','')
print('\n*** Requested to apply the qs predictor at T={} (M={})'.format(T,M))
natoms = 1500
save_path='MLmodel/qs-regression-M{}-T{}'.format(M,Tlabel)
low_thresh_qs=0.00001
ndecimals=10

# check if the model is there
if not os.path.isdir(save_path):
    print('Error: I am looking for the ML model in {}, but I can not find it. If this is the first time, you have to run F_* before this'.format(save_path))
    sys.exit()

# Load the model
predictor = TabularPredictor.load(save_path) 
predictor.persist_models()


# Read the data
Tdir='./output_ML/T{}'.format(T)
dw_df = pd.read_csv('{}/DW_T{}.csv'.format(Tdir,T))
nglass = len(dw_df['conf'].drop_duplicates())
print('\n***Reading the data at T={}\nWe found {} glasses and a total of {} pairs\n\nStarting predictions'.format(T,nglass, len(dw_df)))

# predict the qs
X = dw_df.drop(columns=['i','j','conf','is_dw'])
X = TabularDataset(X).astype(float)
y_pred_by_AI = predictor.predict(dw_df)
y_pred_by_AI = np.power(10, -y_pred_by_AI)
print('The qs has been predicted. Now storing results')

# store the prediction
dw_df['quantum_splitting']=y_pred_by_AI
dw_df = dw_df.sort_values(by='quantum_splitting')
dw_df[['conf','i','j','quantum_splitting']].to_csv('{}/predictedQs_T{}_allpairs.csv'.format(Tdir,T),index=False)
all_qs_df = dw_df.copy()


# load the NEB data for the plot
list_neb_qs=[]
with open('NEB_calculations/T{}/Qs_calculations.txt'.format(T)) as qs_file:
    lines = qs_file.readlines()
    for line in lines:
        qs = line.split()[2]
        list_neb_qs.append(float(qs))
neb_qs=pd.DataFrame({'quantum_splitting':list_neb_qs})



# then I exclude the pairs that I know are NON-dw
list_nondw=[]
with open('NEB_calculations/T{}/NON-DW.txt'.format(T)) as qs_file:
    lines = qs_file.readlines()
    for line in lines:
        conf = int(line.split()[0].strip('Cnf-'))
        i,j = line.split()[1].split('_')
        i = round(float(i),ndecimals)
        j = round(float(j),ndecimals)
        list_nondw.append((conf,i,j))
nondw=pd.DataFrame(list_nondw,columns=['conf','i','j'])
temp_df = all_qs_df.reset_index(drop=True)
temp_df['index'] = temp_df.index
remove_df = temp_df.merge(nondw, how = 'inner' ,indicator=False)
remove_df = remove_df.set_index('index')
all_qs_df = all_qs_df[~all_qs_df.isin(remove_df)].dropna().reset_index()
print('\n*We know that {} of the new pairs are non-dw (from NEB), so we do not need to predict them.\nWe then finish with {} new pairs'.format(len(remove_df),len(all_qs_df)))

# then exclude the pairs for which I run the NEB
list_neb_done=[]
with open('NEB_calculations/T{}/Qs_calculations.txt'.format(T)) as qs_file:
    lines = qs_file.readlines()
    for line in lines:
        conf = int(line.split()[0].strip('Cnf-'))
        i,j = line.split()[1].split('_')
        i = round(float(i),ndecimals)
        j = round(float(j),ndecimals)
        list_neb_done.append((conf,i,j))
neb_done=pd.DataFrame(list_neb_done,columns=['conf','i','j'])
temp_df = all_qs_df.reset_index(drop=True)
temp_df['index'] = temp_df.index
remove_df = temp_df.merge(neb_done, how = 'inner' ,indicator=False)
remove_df = remove_df.set_index('index')
all_qs_df = all_qs_df[~all_qs_df.isin(remove_df)].dropna()
print('\n*For {} of the new pairs we already run the NEB, so we do not need to predict them.\nWe then finish with {} new pairs'.format(len(remove_df),len(all_qs_df)))

# Storing
all_qs_df[['conf','i','j','quantum_splitting']].to_csv('{}/predictedQs_T{}_newpairs.csv'.format(Tdir,T),index=False)
