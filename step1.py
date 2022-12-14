import numpy as np
import glob
import math
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


# This code takes all the available data and retrain the ML model according to the iterative training procedure 


if __name__ == "__main__":
    In_file = myparams.In_file
    In_label = In_file.split('/')[-1]).split('.')[0]
    use_new_calculations = myparams.use_new_calculations
    print('\n*** Requested to train the classifier from {}'.format(In_file))
    ndecimals=myparams.ij_decimals
    if ndecimals>0:
        rounding_error=10**(-1*(ndecimals+1))
    model_path='MLmodel/classification-{}'.format(In_label)

    calc_dirname = 'exact_calculations/{}'.format(In_label) 
    if not os.path.exists(calc_dirname):
        os.makedirs(calc_dirname, exist_ok=True)
    
    # *************
    # First I load the data that the classifier has already used for its training 
    try:
        used_data = pd.read_feather('MLmodel/data-used-by-classifier-{}.feather'.format(In_label))
    except:
        print('First time training the classifier')
        used_data = pd.DataFrame()
    
    # Then I check the exact calculations to see the if new data are available (otherwise the iterative training procedure got stuck)
    list_class_0=[]
    list_class_1=[]
    calculation_dir='./exact_calculations/{}'.format(In_label)
    if not os.path.isfile('{}/NON-DW.txt'.format(calculation_dir)):
        print('\n*(!)* Notice that there are no classification data\n')
        use_new_calculations = False
    else:
        with open('{}/NON-DW.txt'.format(calculation_dir)) as nondw_file:
            lines = nondw_file.readlines()
            for line in lines:
                conf = float(line.split()[0].split('Cnf-')[-1])
                i,j = line.split()[1].split('_')
                i = round(float(i),ndecimals)
                j = round(float(j),ndecimals)
                list_class_0.append((T,conf,i,j))
        with open('{}/Qs_calculations.txt'.format(calculation_dir)) as dw_file:
            lines = dw_file.readlines()
            for line in lines:
                conf = float(line.split()[0].split('Cnf-')[-1])
                i,j = line.split()[1].split('_')
                i = round(float(i),ndecimals)
                j = round(float(j),ndecimals)
                list_class_1.append((T,conf,i,j))
        print('From the NEB results we have {} non-dw and {} dw (with qs)'.format(len(list_class_0),len(list_class_1)))
    
    # I also have to include the pre-training data, which I load now to see if overall we gained data
    try:
        pretrain_df = pd.read_feather('MLmodel/{}'.format(myparams.pretraining_classifier))

        print(pretrain_df.sort_values('Delta_E',ascending=False))
    except:
        print('\nNotice that no pretraining is available')
        pretrain_df = pd.DataFrame()

    
    #************
    # * Check wether or not you should retrain the model
    if(len(pretrain_df)+len(list_class_1)+len(list_class_0))>len(used_data):
        print('\n*****\nThe model was trained using {} data and now we could use:\n\t{} from pretraining (both dw and non-dw)\n\t{} non-dw from NEB\n\t{} dw from NEB'.format(len(used_data),len(pretrain_df),len(list_class_0),len(list_class_1)))
    else:
        print('All the data available have been already used to train the model')
        sys.exit()
    
    # If we are not exited, it means that we have more NEB data to use to retrain the model
    
    if use_new_calculations:        

        # For these new NEB informations, I look for the corresponding input pair 
        # so I need to load the input features for all of them
        try:
            all_pairs_df = pd.read_feather('./Configurations/postprocessing/T{}.feather'.format(Tlabel))
            all_pairs_df.i = all_pairs_df.i.round(ndecimals)
            all_pairs_df.j = all_pairs_df.j.round(ndecimals)
        except:
            print('Error: there are no data prepared')
            sys.exit()

        # split this task between parallel workers
        elements_per_worker=20
        chunks=[list_class_0[i:i + elements_per_worker] for i in range(0, len(list_class_0), elements_per_worker)]
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
                    sys.exit()
                elif len(a)==1:
                    worker_df = pd.concat([worker_df,a])
#                else:
#                    print('\nWARNING: we do not have {}'.format(element))
#                    sys.exit()
            return worker_df
    
        print('collecting info for NEB pairs')
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
        chunks=[list_class_1[i:i + elements_per_worker] for i in range(0, len(list_class_1), elements_per_worker)]
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
    else:
        print('(We are not using data from NEB, but only pretraining)') 
        dw_df = pd.DataFrame()
        non_dw_df = pd.DataFrame()
    
    
    
    # *******
    # add the pretrained data (if any)
    if len(pretrain_df)>0:
        dw_df = pd.concat([dw_df,pretrain_df[pretrain_df['is_dw']>0]])
        dw_df=dw_df.drop_duplicates()
        non_dw_df = pd.concat([non_dw_df,pretrain_df[pretrain_df['is_dw']<1]])
        non_dw_df=non_dw_df.drop_duplicates()
    qs_df = pd.concat([dw_df,non_dw_df])

    # remove useless column
    qs_df = qs_df.drop(columns=['T','i2','j2','index'])
    dw_df = dw_df.drop(columns=['T','i2','j2','index'])
    non_dw_df = non_dw_df.drop(columns=['T','i2','j2','index'])


#    # set isdw col as binary
    qs_df['is_dw']=qs_df['is_dw'].astype(bool)
    dw_df['is_dw']=dw_df['is_dw'].astype(bool)
    non_dw_df['is_dw']=non_dw_df['is_dw'].astype(bool)
    qs_df['Tij']=qs_df['Tij'].astype(int)
    dw_df['Tij']=dw_df['Tij'].astype(int)
    non_dw_df['Tij']=non_dw_df['Tij'].astype(int)
    qs_df['Tji']=qs_df['Tji'].astype(int)
    dw_df['Tji']=dw_df['Tji'].astype(int)
    non_dw_df['Tji']=non_dw_df['Tji'].astype(int)

    
    # This is the new training df that will be stored at the end 
    new_training_df = qs_df.copy()
    if len(new_training_df)<=len(used_data):
        print('\n(!) After removing the duplicates it appears that the number of data has not increased since the last time')
        if os.path.isfile('{}/predictor.pkl'.format(model_path)):
            print('and since the model is already trained, I stop')
            sys.exit()
        else:
            print('but the model is not in {}, so I train anyway'.format(model_path),flush=True)
    
    
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
    if myparams.Fast_dw==True:
        print('We are training in the fast way')
        presets='good_quality_faster_inference_only_refit'
    else:
        presets='high_quality_fast_inference_only_refit'
    #presets='best_quality'
    
    # you can also change the training time
    training_hours=myparams.qs_pred_train_hours
    time_limit = training_hours*60*60

    
    # train
    # * I am excluding KNN because it is problematic
    predictor = TabularPredictor(label='is_dw', path=model_path, eval_metric='accuracy').fit(TabularDataset(training_set.drop(columns=['i','j','conf'])), time_limit=time_limit,  presets=presets,excluded_model_types=['KNN'])
    
    
    # store
    new_training_df.reset_index().drop(columns='index').to_feather('MLmodel/data-used-by-dwclassifier-M{}-T{}.feather'.format(M,Tlabel), compression='zstd')
    training_set.reset_index().drop(columns='index').to_feather('MLmodel/dw-classifier-training-set-M{}-T{}.feather'.format(M,Tlabel), compression='zstd')
    validation_set.reset_index().drop(columns='index').to_feather('MLmodel/dw-classifier-validation-set-M{}-T{}.feather'.format(M,Tlabel), compression='zstd')
