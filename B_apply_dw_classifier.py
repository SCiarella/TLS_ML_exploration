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
        new_df.to_feather('MLmodel/input_features_all_pairs_M{}-T{}.feather'.format(M,Tlabel), compression='zstd')
        sys.exit()
    
    if len(new_df)<len(old_df):
        print('\n***Error: input_features* has lenght {} while I find only {} pairs. This is only possible if you have lost data!'.format(len(old_df),len(new_df)) )
        sys.exit()
    
    
    print('\n\t@@@@ Overall we have {} pairs, while the last time you run this we had {}'.format(len(new_df),len(old_df)))

    
    # * Then I store this df to avoid having to redo it 
    new_df.to_feather('MLmodel/input_features_all_pairs_M{}-T{}.feather'.format(M,Tlabel), compression='zstd')

    # remove the useless columns
    new_df=new_df.drop(columns=['i2','j2','T'])

    # *************
    # (2) remove pairs with a Delta_E which is too large
    new_df = new_df[new_df['Delta_E']<myparams.DeltaEMax]
    print('*-> We decide to keep only the ones with Delta_E<{}, which are {}'.format(myparams.DeltaEMax, len(new_df)))

    
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
    
    npairs = len(new_df)
    chunk_size=1e6
    nchunks = int(npairs/chunk_size)+1
    processed_chunks = []
    print('Splitting the data ({} pairs) in {} parts'.format(npairs,nchunks))
    df_chunks = np.array_split(new_df, nchunks)
    del new_df

    filtered_chunks = []
    for chunk_id, chunk in enumerate(df_chunks):
        print('\n* Classifying part {}/{}'.format(chunk_id+1,nchunks),flush=True)
        df_chunks[chunk_id]['is_dw'] = dwclassifier.predict(chunk.drop(columns=['conf','i','j']))
        # I only keep the predicted dw
        #df_chunks[chunk_id] = df_chunks[chunk_id][df_chunks[chunk_id]['is_dw']>0]
        filtered_chunks.append(df_chunks[chunk_id][df_chunks[chunk_id]['is_dw']>0])
        del df_chunks[chunk_id] 
        print('done in {} sec'.format(time.time() -start))
    timeclass=time.time()-start

    print('\n - Merging results')
    filtered_dw = pd.DataFrame()
    filtered_dw = dd.concat(filtered_chunks)
    filtered_dw = pd.DataFrame(filtered_dw)
    print(filtered_dw)
    
    print('From the {} pairs, only {} are classified as dw (in {} sec = {} sec per pair), so {} are non-dw'.format(npairs, len(filtered_dw), timeclass, timeclass/npairs, npairs-len(filtered_dw)))
    
    filtered_df_name='output_ML/T{}/DW_T{}.csv'.format(T,T)
    filtered_dw.to_csv(filtered_df_name, index=False)
    
