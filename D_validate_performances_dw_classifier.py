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


if len(sys.argv) > 1:
    M = sys.argv[1]
else:
    print('Error because I have not received IN')
    sys.exit()

save_path='MLmodel/dw-classification-M{}'.format(M)
validation_set = pd.read_pickle('MLmodel/dw-classifier-validation-set-M{}.pickle'.format(M))
validation_set = pd.read_pickle('MLmodel/dw-classifier-training-set-M{}.pickle'.format(M))

label= 'is_dw'
#validation_set = validation_set.drop(columns=['i','j','conf'])
# * Convert to float to have optimal performances!
validation_set = TabularDataset(validation_set).astype(float)
y_true_val = validation_set[label]  # values to predict
y_true_val = y_true_val.sort_values(ascending=False)
validation_set= validation_set.sort_values(label,ascending=False)
validation_set_nolab = validation_set.drop(columns=[label])  # delete label column to prove we're not cheating
print(validation_set_nolab)

# Load the model
predictor = TabularPredictor.load(save_path, verbosity=3) 
#predictor.persist_models()
# predict
y_pred_by_AI = predictor.predict(validation_set)
#y_pred_by_AI = predictor.predict(validation_set_nolab)

print('aaaaaaaa\n\n\n\n\n')

perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)


correct_count=0
wrong_count=0
missed_dw=0
count=0
dw_count=0
for pred,real in zip(y_pred_by_AI,y_true_val):
    if pred==real:
        correct_count+=1
    else:
        wrong_count+=1
        if real==1:
            missed_dw+=1
    if real==1:
        dw_count+=1
    count+=1
accuracy = correct_count/len(y_pred_by_AI)
print('Overall the classifier {} has the following performances:\n\t{} accuracy (training set)\n\t{} accuracy (test set)'.format(save_path,accuracy_train,accuracy))

sys.exit()

# *************
# First I load the data that the classifier has already used for its training 
used_data = pd.read_pickle('MLmodel/data-used-by-dwclassifier-M{}.pickle'.format(M))

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
pretrain_df = pd.read_pickle('MLmodel/pretraining-dwclassifier-M{}.pickle'.format(M))
#pretrain_df['Delta_E']=pretrain_df['Delta_E']*1500
#pretrain_df.to_pickle('MLmodel/pretraining-dwclassifier-M{}.pickle'.format(M))
#sys.exit()

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
all_pairs_df = pd.read_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))
all_pairs_df['i'] = all_pairs_df['i'].astype(float)
all_pairs_df['j'] = all_pairs_df['j'].astype(float)
all_pairs_df['T'] = all_pairs_df['T'].astype(float)
all_pairs_df.i = all_pairs_df.i.round(ndecimals)
all_pairs_df.j = all_pairs_df.j.round(ndecimals)


# also it is possible that I have already used them, so I need to check this df
training_df = used_data.copy()
training_df = training_df[training_df['i']!='NotAvail']
training_df['i'] = training_df['i'].astype(float)
training_df['j'] = training_df['j'].astype(float)
training_df['T'] = training_df['T'].astype(float)
training_df.i = training_df.i.round(ndecimals)
training_df.j = training_df.j.round(ndecimals)


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
        # was this element used for training?
        a = training_df[(training_df['T']==T)&(training_df['conf']==conf)&(training_df['i'].between(i-rounding_error,i+rounding_error))&(training_df['j'].between(j-rounding_error,j+rounding_error))]
        if len(a)>1:
            print('Error! multiple correspondences in train')
            sys.exit()
        elif len(a)==1:
            worker_df = pd.concat([worker_df,a])
        else:
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
            else:
                print('we do not have {}'.format(element))
                sys.exit()
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
print('Constructed the database of {} non-dw'.format(len(non_dw_df)))

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
print('Constructed the database of {} dw'.format(len(dw_df)))


# *******
# add the pretrained data (if any)
dw_df = pd.concat([dw_df,pretrain_df[pretrain_df['is_dw']>0]])
dw_df=dw_df.drop_duplicates()
non_dw_df = pd.concat([non_dw_df,pretrain_df[pretrain_df['is_dw']<1]])
non_dw_df=non_dw_df.drop_duplicates()


# This is the new training df that will be stored at the end 
new_training_df = pd.concat([dw_df,non_dw_df])


# Create a balanced subset with same number of dw and non-dw
N = min(len(dw_df),len(non_dw_df))
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


print('From the overall %d data we prepare:\n\t- training set of %d  (half dw and half non-dw) \n\t- validation set of %d  (half dw and half non-dw)\n\n'%(len(new_training_df),len(training_set),len(validation_set) ),flush=True)


print(training_set)
print(validation_set)



# **************   TRAIN
#   Notice that autogluon offer different 'presets' option to maximize precision vs data-usage vs time
#   if you are not satisfied with the results here, you can try different 'presets' option or build your own
# check which one to use
presets='high_quality_fast_inference_only_refit'
#presets='best_quality'
#presets='good_quality_faster_inference_only_refit'

# you can also change the training time
training_hours=0.02
time_limit = training_hours*60*60

# remove the info not needed
training_set.drop(columns=['i','j','conf'])

# train
predictor = TabularPredictor(label='is_dw', path='MLmodel/dw-classification-M{}'.format(M), eval_metric='accuracy').fit(training_set, time_limit=time_limit,  presets=presets)


# store
new_training_df.to_pickle('MLmodel/data-used-by-dwclassifier-M{}.pickle'.format(M))
training_set.to_pickle('MLmodel/dw-classifier-training-set-M{}.pickle'.format(M))
validation_set.to_pickle('MLmodel/dw-classifier-validation-set-M{}.pickle'.format(M))
