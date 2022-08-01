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
    model_path='MLmodel/qs-regression-M{}'.format(M)
    
    # *************
    # First I load the data that the predictor has already used for its training 
    try:
        used_data = pd.read_pickle('MLmodel/data-used-by-qspredictor-M{}.pickle'.format(M))
    except:
        print('First time training the classifier')
        used_data = pd.DataFrame()
    
    # Then I check the NEB calculations to see the what new data are available 
    list_neb_qs=[]
    list_T = glob.glob('NEB_calculations/*')
    for Tdir in list_T:
        T=float(Tdir.split('/T')[-1])
        with open('{}/Qs_calculations.txt'.format(Tdir)) as qs_file:
            lines = qs_file.readlines()
            for line in lines:
                conf = line.split()[0]
                i,j = line.split()[1].split('_')
                i = round(float(i),ndecimals)
                j = round(float(j),ndecimals)
                qs = line.split()[2]
                list_neb_qs.append((T,conf,i,j,qs))
    print('From the NEB results we have {} pairs for which we know the qs'.format(len(list_neb_qs)))
    
    
    # then load the info about all the pairs
    pairs_df = pd.read_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))
    # and format in the correct way
    pairs_df['i'] = pairs_df['i'].astype(float)
    pairs_df['j'] = pairs_df['j'].astype(float)
    pairs_df = pairs_df.round(decimals=10)
    
    
    # I also have to include the pre-training data, which I load now to see if overall we gained data
    try:
        pretrain_df = pd.read_pickle('MLmodel/pretraining-qs-regression-M{}.pickle'.format(M))
    except:
        print('\nNotice that no pretraining is available')
        pretrain_df = pd.DataFrame()
    
    #************
    # * Check wether or not you should retrain the model
    if(len(pretrain_df)+len(list_neb_qs))>len(used_data):
        print('\n*****\nThe model was trained using {} data and now we could use:\n\t{} from pretraining \n\t{} from NEB'.format(len(used_data),len(pretrain_df),len(list_neb_qs)))
    else:
        print('All the data available have been already used to train the model')
        sys.exit()
    
    # If we are not exited, it means that we have more qs data to use to retrain the model
    
    
    # split this task between parallel workers
    elements_per_worker=100
    chunks=[list_neb_qs[i:i + elements_per_worker] for i in range(0, len(list_neb_qs), elements_per_worker)]
    n_chunks = len(chunks)
    print('We are going to submit {} chunks to get the data\n'.format(n_chunks))
    
    
    
    def process_chunk(chunk):
        worker_df=pd.DataFrame()
        # I search for the given configuration
        for element in chunk:
            T,conf,i,j,qs = element
            a = pairs_df[(pairs_df['T']==T)&(pairs_df['conf']==conf)&(pairs_df['i'].between(i-rounding_error,i+rounding_error))&(pairs_df['j'].between(j-rounding_error,j+rounding_error))]
            if len(a)>1:
                print('Error! multiple correspondences in dw')
                sys.exit()
            elif len(a)==1:
                a['quantum_splitting']=qs
                worker_df = pd.concat([worker_df,a])
           # else:
           #     print('Error: we do not have {}'.format(element))
           #     sys.exit()
        return worker_df
            
    # Initialize the pool
    pool = mp.Pool(mp.cpu_count())
    # *** RUN THE PARALLEL FUNCTION
    results = pool.map(process_chunk, [chunk for chunk in chunks] )
    pool.close()
    # and add all the new df to the final one
    qs_df=pd.DataFrame()
    missed_dw=0
    for df_chunk in results:
        qs_df= pd.concat([qs_df,df_chunk])
    print('From the NEB calculations I constructed a database of {} pairs for which I have the input informations.'.format(len(qs_df)))
    
    
    # *******
    # add the pretrained data (if any)
    if len(pretrain_df)>0:
        qs_df = pd.concat([qs_df,pretrain_df])
        qs_df = qs_df.drop_duplicates()
        qs_df = qs_df.reset_index(drop=True)


    # Check that the different temperatures are balanced in the data
    T_list = qs_df['T'].unique()
    print('\n')
    ndata_for_each_T=len(qs_df)
    for T in T_list:
        nd=len(qs_df[qs_df['T']==T])
        print('We have {} data at T={}'.format(nd,T))
        if nd<ndata_for_each_T:
            ndata_for_each_T=nd
    print('so we only keep {} for each different T to mantain a balanced dataset'.format(ndata_for_each_T))
    balanced_df=pd.DataFrame()
    for T in T_list:
        balanced_df=pd.concat([balanced_df, (qs_df[qs_df['T']==T]).sample(n=ndata_for_each_T)])
    qs_df=balanced_df


    
    # This is the new training df that will be stored at the end 
    new_training_df = qs_df.copy()
    if len(new_training_df)<=len(used_data):
        print('\n(!) After removing the duplicates it appears that the number of data has not increased since the last time')
        if os.path.isfile('{}/predictor.pkl'.format(model_path)):
            print('and since the model is already trained, I stop')
            sys.exit()
        else:
            print('but the model is not in {}, so I train anyway'.format(model_path),flush=True)
    
    
    # convert to float
    new_training_df['Delta_E'] = new_training_df['Delta_E'].astype(float)
    new_training_df['quantum_splitting'] = new_training_df['quantum_splitting'].astype(float)
    for mi in range(int(M)):
        new_training_df['displacement_{}'.format(mi)] = new_training_df['displacement_{}'.format(mi)].astype(float)
    

    # ************
    # *** I do (-1) log of the data such that the values are closer and their weight is more balanced in the fitness
    new_training_df['10tominusquantum_splitting'] = new_training_df['quantum_splitting'].apply(lambda x: -np.log10(x))
    
    
    # Pick 10percent of this pairs for validation
    N = len(new_training_df)
    Nval = int(0.1*N)
    Ntrain = N -Nval
    # shuffle
    new_training_df=new_training_df.sample(frac=1, random_state=20, ignore_index=True)
    # and slice
    training_set = new_training_df.iloc[:Ntrain]
    validation_set = new_training_df.iloc[Ntrain:N]
    
    print('\nFrom the overall %d data we prepare:\n\t- training set of %d\n\t- validation set of %d\n\n'%(len(new_training_df),len(training_set),len(validation_set) ),flush=True)

    
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
    
    # and define a weight=1/qs in order to increase the precision for the small qs 
    training_set['weights'] = (training_set['quantum_splitting']) ** (-1)
    
    # train
    # * I am excluding KNN because it is problematic
    # * Convert to float to have optimal performances!
    predictor = TabularPredictor(label='10tominusquantum_splitting', path=model_path, eval_metric='mean_squared_error', sample_weight='weights' , weight_evaluation=True).fit(TabularDataset(training_set.drop(columns=['i','j','conf','quantum_splitting'])).astype(float), time_limit=time_limit,  presets=presets,excluded_model_types=['KNN'])
    
    
    # store
    new_training_df.to_pickle('MLmodel/data-used-by-qspredictor-M{}.pickle'.format(M))
    training_set.to_pickle('MLmodel/qs-prediction-training-set-M{}.pickle'.format(M))
    validation_set.to_pickle('MLmodel/qs-prediction-validation-set-M{}.pickle'.format(M))
