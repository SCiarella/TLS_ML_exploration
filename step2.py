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

def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

if __name__ == "__main__":
    In_file = myparams.In_file
    In_label = In_file.split('/')[-1].split('.')[0]
    print('\n*** Requested to apply the classifier to all the pairs in {}'.format(In_file))

    ensure_dir('output_ML/{}/'.format(In_label))
    
    # *************
    # (1) Load the preprocessed data 
    new_df = pd.read_feather('IN_data/{}'.format(myparams.In_file))
    
    print('\n\t@@@@ Overall we have {} pairs'.format(len(new_df)))

    # *************
    # (2) (optionally) remove pairs with a classic energy splitting which is too large
    new_df = new_df[new_df[r'$\Delta E$']<0.1]
    print('*-> We decide to keep only the ones with Delta_E<{}, which are {}'.format(0.1, len(new_df)))

    
    # *************
    # (3) apply the dw Filter 
    start= time.time()
    classifier_save_path='MLmodel/classification-{}'.format(In_label)
    # check if the model is there
    if not os.path.isdir(classifier_save_path):
        print('Error: I am looking for the classifier in {}, but I can not find it. You probably have to run step1 before this'.format(classifier_save_path))
        sys.exit()
    else:
        print('\nUsing the DW filter trained in {}'.format(classifier_save_path))
    
    print('\n* Classifier loading',flush=True)
    dwclassifier = TabularPredictor.load(classifier_save_path) 
    
    npairs = len(new_df)
    chunk_size=1e6
    nchunks = int(npairs/chunk_size)+1
    processed_chunks = []
    print('Temporarily splitting the data ({} pairs) in {} parts'.format(npairs,nchunks))
    df_chunks = np.array_split(new_df, nchunks)
    del new_df
    print('Classification starting:',flush=True)

    filtered_chunks = []
    filtered_dw = pd.DataFrame()
    for chunk_id, chunk in enumerate(df_chunks):
        print('\n* Classifying part {}/{}'.format(chunk_id+1,nchunks),flush=True)
        df_chunks[chunk_id]['is_dw'] = dwclassifier.predict(chunk.drop(columns=['conf','i','j','i2','j2','T']))
        #df_chunks[chunk_id]['is_dw'] = dwclassifier.predict(chunk.drop(columns=['conf','i','j']))
        # I only keep the predicted dw
        filtered_dw = pd.concat([filtered_dw,df_chunks[chunk_id][df_chunks[chunk_id]['is_dw']>0]])
        print('done in {} sec (collected up to {} dw) '.format(time.time() -start, len(filtered_dw)))
    timeclass=time.time()-start

    print('From the {} pairs, only {} are classified as dw (in {} sec = {} sec per pair), so {} are non-dw.'.format(npairs, len(filtered_dw), timeclass, timeclass/npairs, npairs-len(filtered_dw)))
    
    filtered_df_name='output_ML/T{}/DW_T{}.csv'.format(T,T)
    filtered_dw.to_csv(filtered_df_name, index=False)
