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
import multiprocess as mp
import myparams


# This code takes all the available data (results from the NEB) and if they are more than the data that we already used to train the model, we retrain it 


if __name__ == "__main__":
    M = myparams.M
    ndecimals=10
    rounding_error=10**(-1*(ndecimals+1))
    model_path='MLmodel/dw-classification-M{}'.format(M)
    
    # *************
    # First I load the data that the classifier has already used for its training 
    try:
        used_data = pd.read_pickle('MLmodel/data-used-by-dwclassifier-M{}.pickle'.format(M))
    except:
        print('First time training the classifier')
        used_data = pd.DataFrame()
    
    # Then I check the NEB calculations to see the what new data are available 
    list_neb_nondw=[]
    list_neb_dw=[]
    list_T = glob.glob('NEB_calculations/*')
    for Tdir in list_T:
        T=float(Tdir.split('/T')[-1])
        with open('{}/NON-DW.txt'.format(Tdir)) as nondw_file:
            lines = nondw_file.readlines()
            for line in lines:
                conf = line.split()[0]
                i,j = line.split()[1].split('_')
                i = round(float(i),ndecimals)
                j = round(float(j),ndecimals)
                list_neb_nondw.append((T,conf,i,j))
        with open('{}/Qs_calculations.txt'.format(Tdir)) as dw_file:
            lines = dw_file.readlines()
            for line in lines:
                conf = line.split()[0]
                i,j = line.split()[1].split('_')
                i = round(float(i),ndecimals)
                j = round(float(j),ndecimals)
                list_neb_dw.append((T,conf,i,j))
    print('From the NEB results we have {} non-dw and {} dw (with qs)'.format(len(list_neb_nondw),len(list_neb_dw)))
    
    # I also have to include the pre-training data, which I load now to see if overall we gained data
    try:
        pretrain_df = pd.read_pickle('MLmodel/pretraining-dwclassifier-M{}.pickle'.format(M))
    except:
        print('\nNotice that no pretraining is available')
        pretrain_df = pd.DataFrame()
    
    #************
    # * Check wether or not you should retrain the model
    if(len(pretrain_df)+len(list_neb_dw)+len(list_neb_nondw))>len(used_data):
        print('\n*****\nThe model was trained using {} data and now we could use:\n\t{} from pretraining (both dw and non-dw)\n\t{} non-dw from NEB\n\t{} dw from NEB'.format(len(used_data),len(pretrain_df),len(list_neb_nondw),len(list_neb_dw)))
    else:
        print('All the data available have been already used to train the model')
        sys.exit()
    
    # If we are not exited, it means that we have more NEB data to use to retrain the model
    
    # For these new NEB informations, I look for the corresponding input pair 
    # so I need to load the input features for all of them
    try:
        all_pairs_df = pd.read_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))
    except:
        print('Error: there are no data prepared, so you have to run B_ first.')
        sys.exit()
    all_pairs_df['i'] = all_pairs_df['i'].astype(float)
    all_pairs_df['j'] = all_pairs_df['j'].astype(float)
    all_pairs_df['T'] = all_pairs_df['T'].astype(float)
    all_pairs_df.i = all_pairs_df.i.round(ndecimals)
    all_pairs_df.j = all_pairs_df.j.round(ndecimals)
    
    
    # split this task between parallel workers
    elements_per_worker=20
    chunks=[list_neb_nondw[i:i + elements_per_worker] for i in range(0, len(list_neb_nondw), elements_per_worker)]
    n_chunks = len(chunks)
    print('We are going to submit {} chunks for the non dw \n'.format(n_chunks))
    
    
    def process_chunk(chunk):
        worker_df=pd.DataFrame()
        # I search for the given configuration
        for element in chunk:
            T,conf,i,j = element
            # do we have it ?
            a = all_pairs_df[(all_pairs_df['T']==T)&(all_pairs_df['conf']==conf)&(all_pairs_df['i'].between(i-rounding_error,i+rounding_error))&(all_pairs_df['j'].between(j-rounding_error,j+rounding_error))]
            if len(a)>1:
                print('Error! multiple correspondences for {}'.format(element))
                with pd.option_context('display.float_format', '{:0.20f}'.format):
                    print(a)
                    print(a[['i','j']])
                sys.exit()
            elif len(a)==1:
                worker_df = pd.concat([worker_df,a])
    #        else:
    #            print('WARNING: we do not have {}'.format(element))
        return worker_df
    
            
    # Initialize the pool
    pool = mp.Pool(mp.cpu_count())
    # *** RUN THE PARALLEL FUNCTION
    results = pool.map(process_chunk, [chunk for chunk in chunks] )
    pool.close()
    # and add all the new df to the final one
    non_dw_df=pd.DataFrame()
    for df_chunk in results:
        non_dw_df = pd.concat([non_dw_df,df_chunk])
    non_dw_df['is_dw']=0
    print('Constructed the database of {} non-dw from the new pairs'.format(len(non_dw_df)))
    
    # *** now get the dw using the same function
    chunks=[list_neb_dw[i:i + elements_per_worker] for i in range(0, len(list_neb_dw), elements_per_worker)]
    n_chunks = len(chunks)
    print('We are going to submit {} chunks for the dw \n'.format(n_chunks))
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(process_chunk, [chunk for chunk in chunks] )
    pool.close()
    dw_df=pd.DataFrame()
    for df_chunk in results:
        dw_df = pd.concat([dw_df,df_chunk])
    dw_df['is_dw']=1
    print('Constructed the database of {} dw from the new pairs'.format(len(dw_df)))
    
    
    
    # *******
    # add the pretrained data (if any)
    if len(pretrain_df)>0:
        dw_df = pd.concat([dw_df,pretrain_df[pretrain_df['is_dw']>0]])
        dw_df=dw_df.drop_duplicates()
        non_dw_df = pd.concat([non_dw_df,pretrain_df[pretrain_df['is_dw']<1]])
        non_dw_df=non_dw_df.drop_duplicates()
    
    
    # This is the new training df that will be stored at the end 
    new_training_df = pd.concat([dw_df,non_dw_df])
    if len(new_training_df)<=len(used_data):
        print('\n(!) After removing the duplicates it appears that the number of data has not increased since the last time')
        if os.path.isfile('{}/predictor.pkl'.format(model_path)):
            print('and since the model is already trained, I stop')
            sys.exit()
        else:
            print('but the model is not in {}, so I train anyway'.format(model_path),flush=True)
    
    # convert to float
    new_training_df['Delta_E'] = new_training_df['Delta_E'].astype(float)
    for mi in range(int(M)):
        new_training_df['displacement_{}'.format(mi)] = new_training_df['displacement_{}'.format(mi)].astype(float)
    
    # Create a balanced subset with same number of dw and non-dw
    N = min(len(dw_df),len(non_dw_df))
    print('Having {} dw and {} non-dw, we select only {} of each for the classifier'.format(len(dw_df),len(non_dw_df),N))
    # and pick 10percent apart for validation
    Nval = int(0.1*N)
    Ntrain = N -Nval
    # shuffle
    dw_df=dw_df.sample(frac=1, random_state=20, ignore_index=True)
    non_dw_df=non_dw_df.sample(frac=1, random_state=20, ignore_index=True)
    # and slice
    training_set = pd.concat([dw_df.iloc[:Ntrain],non_dw_df.iloc[:Ntrain]])
    validation_set = pd.concat([dw_df.iloc[Ntrain:N],non_dw_df.iloc[Ntrain:N]])
    # and reshuffle
    training_set = training_set.sample(frac=1, random_state=20, ignore_index=True)
    validation_set = validation_set.sample(frac=1, random_state=20, ignore_index=True)
    
    
    print('\nFrom the overall %d data we prepare:\n\t- training set of %d  (half dw and half non-dw) \n\t- validation set of %d  (half dw and half non-dw)\n\n'%(len(new_training_df),len(training_set),len(validation_set) ),flush=True)
    
    
    
    # **************   TRAIN
    #   Notice that autogluon offer different 'presets' option to maximize precision vs data-usage vs time
    #   if you are not satisfied with the results here, you can try different 'presets' option or build your own
    # check which one to use
    presets='high_quality_fast_inference_only_refit'
    #presets='best_quality'
    #presets='good_quality_faster_inference_only_refit'
    
    # you can also change the training time
    training_hours=myparams.qs_pred_train_hours
    time_limit = training_hours*60*60
    
    # train
    # * I am excluding KNN because it is problematic
    # * Convert to float to have optimal performances!
    predictor = TabularPredictor(label='is_dw', path=model_path, eval_metric='accuracy').fit(TabularDataset(training_set.drop(columns=['i','j','conf'])).astype(float), time_limit=time_limit,  presets=presets,excluded_model_types=['KNN'])
    
    
    # store
    new_training_df.to_pickle('MLmodel/data-used-by-dwclassifier-M{}.pickle'.format(M))
    training_set.to_pickle('MLmodel/dw-classifier-training-set-M{}.pickle'.format(M))
    validation_set.to_pickle('MLmodel/dw-classifier-validation-set-M{}.pickle'.format(M))
