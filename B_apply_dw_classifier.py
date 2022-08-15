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
import multiprocess as mp
import myparams
import dask.dataframe as dd

# This code runs the dw vs non-dw classifier over all the pairs 


# The absolute position is not interesting, so I only report the distance from particle 0
# notice that you have to evalaute the distance using PBC !
Lhalf= 0.5 * 11.447142426
def PBC_dist(x):
    while x > Lhalf:
        x-=2*Lhalf
    while x < -Lhalf:
        x+=2*Lhalf
    return x

if __name__ == "__main__":
    M = myparams.M
    T = myparams.T
    Tlabel = str(T).replace('.','')
    print('\n*** Requested to apply the dw classifier at T={} (M={})'.format(T,M))
    
    # *************
    # (1) Load the preprocessed data 
    new_df=pd.read_feather('./Configurations/postprocessing/T{}.feather'.format(Tlabel))
    
    # load the df containing all the pairs that I found last time
    try:
        old_df = pd.read_feather('MLmodel/input_features_all_pairs_M{}-T{}.feather'.format(M,Tlabel))
    except:
        print('First time running this code. Have you already trained the classifier using C_* ? If yes you can re-execute this')
        new_df['i'] = new_df['i'].astype(float)
        new_df['j'] = new_df['j'].astype(float)
        new_df['T'] = new_df['T'].astype(float)
        new_df['Delta_E'] = new_df['Delta_E'].astype(float)
        new_df['PR'] = new_df['PR'].astype(float)
        new_df['total_displacement'] = new_df['total_displacement'].astype(float)
        new_df['conf'] = new_df['conf'].astype(str)
        for mi in range(int(M)):
            new_df['displacement_{}'.format(mi)] = new_df['displacement_{}'.format(mi)].astype(float) 
        new_df.to_feather('MLmodel/input_features_all_pairs_M{}-T{}.feather'.format(M,Tlabel), compression='zstd')
        sys.exit()
    
    if len(new_df)<len(old_df):
        print('\n***Error: input_features* has lenght {} while I find only {} pairs. This is only possible if you have lost data!'.format(len(old_df),len(new_df)) )
        sys.exit()
    
    # Convert the data to the correct types
    print('\nConverting data types')
    new_df['i'] = new_df['i'].astype(float)
    new_df['j'] = new_df['j'].astype(float)
    new_df['T'] = new_df['T'].astype(float)
    new_df['Delta_E'] = new_df['Delta_E'].astype(float)
    new_df['PR'] = new_df['PR'].astype(float)
    new_df['total_displacement'] = new_df['total_displacement'].astype(float)
    new_df['conf'] = new_df['conf'].astype(str)
    old_df['i'] = old_df['i'].astype(float)
    old_df['j'] = old_df['j'].astype(float)
    old_df['T'] = old_df['T'].astype(float)
    old_df['conf'] = old_df['conf'].astype(str)
    old_df['PR'] = old_df['PR'].astype(float)
    old_df['total_displacement'] = old_df['total_displacement'].astype(float)
    old_df['Delta_E'] = old_df['Delta_E'].astype(float)
    for mi in range(int(M)):
        new_df['displacement_{}'.format(mi)] = new_df['displacement_{}'.format(mi)].astype(float) 
        old_df['displacement_{}'.format(mi)] = old_df['displacement_{}'.format(mi)].astype(float) 
    
    # check which data are shared and which one are new
    print('\nCross-check and merging')
    #used_df = pd.merge(old_df, new_df) 
    used_df = dd.merge(old_df, new_df) 
    Nnew=len(new_df)-len(used_df)
    
    print('\n\t@@@@ Overall we have {} pairs, of which {} are new from the last time you run this'.format(len(new_df),Nnew))
    
    # * Then I store this df to avoid having to redo it 
    new_df.to_feather('MLmodel/input_features_all_pairs_M{}-T{}.feather'.format(M,Tlabel), compression='zstd')

    # remove the useless columns
    new_df=new_df.drop(columns=['i2','j2','T'])

    
    # *************
    # (3) apply the dw Filter 
    start= time.time()
    classifier_save_path = 'MLmodel/dw-classification-M{}-T{}'.format(M,Tlabel)
    # check if the model is there
    if not os.path.isdir(classifier_save_path):
        print('Error: I am looking for the classifier in {}, but I can not find it. If this is the first time, you have to run C_* before this'.format(classifier_save_path))
        sys.exit()
    else:
        print('\nUsing the DW filter trained in {}'.format(classifier_save_path))
    
    print('\n* Classifier loading',flush=True)
    dwclassifier = TabularPredictor.load(classifier_save_path) 
    
    print('\n* Classifier starts',flush=True)
    new_df['is_dw'] = dwclassifier.predict(new_df.drop(columns=['conf','i','j']))
    timeclass=time.time() -start

    print(new_df)
    
    filtered_dw = new_df[ new_df['is_dw']>0 ] 
    filtered_non_dw = new_df[ new_df['is_dw']<1 ] 
    print('From the {} pairs, only {} are classified as dw (in {} sec = {} sec per pair), so {} are non-dw'.format(len(new_df), len(filtered_dw), timeclass, timeclass/len(new_df), len(filtered_non_dw)))
    
    filtered_df_name='output_ML/T{}/DW_T{}.csv'.format(T,T)
    filtered_dw.to_csv(filtered_df_name, index=False)
    
    filtered_non_dw_name='output_ML/T{}/nonDW_T{}.feather'.format(T,T)
    filtered_non_dw.reset_index().to_feather(filtered_non_dw_name, compression='zstd')
