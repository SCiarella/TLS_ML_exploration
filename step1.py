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


# This code takes all the available data and retrain the ML model [THE FILTER] according to the iterative training procedure 


if __name__ == "__main__":
    In_file = myparams.In_file
    In_label = In_file.split('/')[-1].split('.')[0]
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
    
    # Then I check the exact calculations to see the if new data are available 
    calculation_dir='./exact_calculations/{}'.format(In_label)
    if not os.path.isfile('{}/{}'.format(calculation_dir,myparams.calculations_classifier)):
        print('\n*(!)* Notice that there are no classification data\n')
        use_new_calculations = False
        class_0_pairs = pd.DataFrame()
        class_1_pairs = pd.DataFrame()
    else:
        use_new_calculations = True
        class_0_pairs = pd.read_csv('{}/{}'.format(calculation_dir,myparams.calculations_classifier), index_col=0)
        class_1_pairs = pd.read_csv('{}/{}'.format(calculation_dir,myparams.calculations_predictor), index_col=0)[['conf','i','j']]
        print('From the calculation results we have {} class-0 and {} class-1'.format(len(class_0_pairs),len(class_1_pairs)))


    
    # I also have to include the pre-training data, which I now load to see if overall there are more data than the previous iteration (as expected)
    if os.path.isfile('MLmodel/{}'.format(myparams.pretraining_classifier)):
        pretrain_df = pd.read_feather('MLmodel/{}'.format(myparams.pretraining_classifier))
    else:
        print('\nNotice that no pretraining is available')
        pretrain_df = pd.DataFrame()

    
    #************
    # * Check if the model needs to be retrained (you should have collected new data)
    if(len(pretrain_df)+len(class_0_pairs)+len(class_1_pairs))>len(used_data):
        print('\n*****\nThe model was trained using {} data and now we could use:\n\t{} from pretraining (both classes)\n\t{} calculated class-0\n\t{} calculated class-1'.format(len(used_data),len(pretrain_df),len(class_0_pairs),len(class_1_pairs)))
    else:
        print('All the data available have been already used to train the model')
        sys.exit()
    
    # If the program did not terminate, it means that there are new data and it is worth to retrain the model
    
    if use_new_calculations:        

        # For the new target_feature information, we look for the corresponding input pair 
        # so we need to load the input features for all of them
        try:
            all_pairs_df = pd.read_feather('In_data/{}'.format(myparams.In_file))
            if ndecimals>0:
                all_pairs_df.i = all_pairs_df.i.round(ndecimals)
                all_pairs_df.j = all_pairs_df.j.round(ndecimals)
        except:
            print('Error: there are no data prepared')
            sys.exit()

        # split this task between parallel workers
        # Notice that we are first processing the bad pairs (class 0)
        elements_per_worker=20
        chunks=[class_0_pairs.iloc[i:i + elements_per_worker] for i in range(0, len(class_0_pairs), elements_per_worker)]
        n_chunks = len(chunks)
        print('\nWe are going to submit {} chunks for the bad pairs'.format(n_chunks))
        
        
        def process_chunk(chunk):
            worker_df=pd.DataFrame()
            # I search for the given configuration
            for index, row in chunk.iterrows():
                conf = row['conf']
                i = row['i']
                j = row['j']
                # do we have it ?
                if ndecimals>0:
                    a = all_pairs_df[(all_pairs_df['conf']==conf)&(all_pairs_df['i'].between(i-rounding_error,i+rounding_error))&(all_pairs_df['j'].between(j-rounding_error,j+rounding_error))]
                else:
                    a = all_pairs_df[(all_pairs_df['conf']==conf)&(all_pairs_df['i']==i)&(all_pairs_df['j']==j)]
                if len(a)>1:
                    print('Error! multiple correspondences for {}'.format(element))
                    sys.exit()
                elif len(a)==1:
                    worker_df = pd.concat([worker_df,a])
            return worker_df
    
        print('collecting info for bad pairs')
        # Initialize the pool
        pool = mp.Pool(mp.cpu_count())
        # *** RUN THE PARALLEL FUNCTION
        results = pool.map(process_chunk, [chunk for chunk in chunks] )
        pool.close()
        # and add all the new df to the final one
        badpairs_df=pd.DataFrame()
        for df_chunk in results:
            badpairs_df = pd.concat([badpairs_df,df_chunk])
        badpairs_df['class']=0
        print('Constructed the database of {} bad pairs, from the new data'.format(len(badpairs_df)))
        
        # *** now get the good pairs using the same routine
        chunks=[class_1_pairs.iloc[i:i + elements_per_worker] for i in range(0, len(class_1_pairs), elements_per_worker)]
        n_chunks = len(chunks)
        print('\nWe are going to submit {} chunks for the good data'.format(n_chunks))
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(process_chunk, [chunk for chunk in chunks] )
        pool.close()
        goodpairs_df=pd.DataFrame()
        for df_chunk in results:
            goodpairs_df = pd.concat([goodpairs_df,df_chunk])
        goodpairs_df['class']=1
        print('Constructed the database of {} good pairs, from the new data'.format(len(goodpairs_df)))
    else:
        print('(We are not using data from calculations, but only the pretraining set)') 
        goodpairs_df = pd.DataFrame()
        badpairs_df = pd.DataFrame()
    
    
    # *******
    # add the pretrained data (if any)
    if len(pretrain_df)>0:
        goodpairs_df = pd.concat([goodpairs_df,pretrain_df[pretrain_df['class']>0]])
        goodpairs_df=goodpairs_df.drop_duplicates()
        badpairs_df = pd.concat([badpairs_df,pretrain_df[pretrain_df['class']<1]])
        badpairs_df=badpairs_df.drop_duplicates()
    qs_df = pd.concat([goodpairs_df,badpairs_df])


    # set class column as binary
    qs_df['class']=qs_df['class'].astype(bool)
    goodpairs_df['class']=goodpairs_df['class'].astype(bool)
    badpairs_df['class']=badpairs_df['class'].astype(bool)

    
    # This is the new training df that will be stored at the end 
    new_training_df = qs_df.copy()
    if len(new_training_df)<=len(used_data):
        print('\n(!) After removing the duplicates it appears that the number of data has not increased since the last time')
        if os.path.isfile('{}/predictor.pkl'.format(model_path)):
            print('and since the model is already trained, I stop')
            sys.exit()
        else:
            print('but the model is not in {}, so I train anyway'.format(model_path),flush=True)
    
    
    # Create a balanced subset with same number of good and bad pairs
    N = min(len(goodpairs_df),len(badpairs_df))
    print('Having {} good and {} bad pairs, we select only {} of each, to balance the classifier'.format(len(goodpairs_df),len(badpairs_df),N))
    # and split a part of data for validation
    Nval = int(myparams.validation_split*N)
    Ntrain = N -Nval
    # shuffle
    goodpairs_df=goodpairs_df.sample(frac=1, random_state=20, ignore_index=True)
    badpairs_df=badpairs_df.sample(frac=1, random_state=20, ignore_index=True)
    # and slice
    training_set = pd.concat([goodpairs_df.iloc[:Ntrain],badpairs_df.iloc[:Ntrain]])
    validation_set = pd.concat([goodpairs_df.iloc[Ntrain:N],badpairs_df.iloc[Ntrain:N]])
    # and reshuffle
    training_set = training_set.sample(frac=1, random_state=20, ignore_index=True)
    validation_set = validation_set.sample(frac=1, random_state=20, ignore_index=True)
    
    
    print('\nFrom the overall %d data we prepare:\n\t- training set of %d  (half good and half bad) \n\t- validation set of %d  (half good and half bad)\n\n'%(len(new_training_df),len(training_set),len(validation_set) ),flush=True)
    print(training_set)

    
    
    # **************   TRAIN
    #   Notice that autogluon offer different 'presets' option to maximize precision vs data-usage vs time
    #   if you are not satisfied with the results here, you can try different 'presets' option or build your own
    # check which one to use
    if myparams.Fast_class==True:
        print('We are training in the fast way')
        presets='good_quality_faster_inference_only_refit'
    else:
        presets='high_quality_fast_inference_only_refit'
    
    # you can also change the training time
    training_hours=myparams.class_train_hours
    time_limit = training_hours*60*60

    
    # train
    # * I am excluding the KNN model because it is problematic
    predictor = TabularPredictor(label='class', path=model_path, eval_metric='accuracy').fit(TabularDataset(training_set.drop(columns=['i','j','conf'])), time_limit=time_limit,  presets=presets,excluded_model_types=['KNN'])
    
    
    # store
    new_training_df.reset_index().drop(columns='index').to_feather('MLmodel/data-used-by-classifier-{}.feather'.format(In_label), compression='zstd')
    training_set.reset_index().drop(columns='index').to_feather('MLmodel/classifier-training-set-{}.feather'.format(In_label), compression='zstd')
    validation_set.reset_index().drop(columns='index').to_feather('MLmodel/classifier-validation-set-{}.feather'.format(In_label), compression='zstd')
