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


# We need the desired M for the target df
try:
    with open("M_val.txt") as f:
        M=int(f.readlines()[0].strip('\n'))
        print('M={}'.format(M))
except Exception as error:
    print('Error: {}'.format(Exception))
    sys.exit()

ndecimals=10
rounding_error=10**(-1*(ndecimals+1))
model_path='MLmodel/qs-regression-M{}'.format(M)

# *************
# First I load the data that the predictor has already used for its training 
used_data = pd.read_pickle('MLmodel/data-used-by-qspredictor-M{}.pickle'.format(M))

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

# then I check which pair is also classified as dw
dw_df = pd.DataFrame()
list_T = glob.glob('output_ML/T*')
for Tdir in list_T:
    T=float(Tdir.split('/T')[-1])
    dw_df= pd.concat([dw_df, pd.read_csv('{}/DW_T{}.csv'.format(Tdir,T))])
print('And we have a list of {} pairs classified as dw'.format(len(dw_df)))

## get also a list of non_dw
#nondw_df = pd.DataFrame()
#list_T = glob.glob('output_ML/T*')
#for Tdir in list_T:
#    T=float(Tdir.split('/T')[-1])
##    nondw_df=  pd.read_csv('{}/nonDW_T{}.csv'.format(Tdir,T)).drop(columns='Unnamed: 0')
##    nondw_df.to_csv('{}/nonDW_T{}.csv'.format(Tdir,T),index=False)
#    nondw_df= pd.concat([nondw_df, pd.read_csv('{}/nonDW_T{}.csv'.format(Tdir,T))])
#print('(and {} non-dw)'.format(len(nondw_df)))

# check if you have to classify again
if len(dw_df)<len(list_neb_qs):
    print('\nError: we have more NEB for qs that classified dw, so you should be running the classifier (B_) before this')
    sys.exit()


# I also have to include the pre-training data, which I load now to see if overall we gained data
pretrain_df = pd.read_pickle('MLmodel/pretraining-qs-regression-M{}.pickle'.format(M))

#************
# * Check wether or not you should retrain the model
if(len(pretrain_df)+len(list_neb_qs))>len(used_data):
    print('\n*****\nThe model was trained using {} data and now we could use:\n\t{} from pretraining \n\t{} from NEB'.format(len(used_data),len(pretrain_df),len(list_neb_qs)))
else:
    print('All the data available have been already used to train the model')
    sys.exit()

# If we are not exited, it means that we have more qs data to use to retrain the model

# For these new qs informations, I look for the corresponding input pair, from the output of the classifier 
#dw_df['i'] = dw_df['i'].astype(float)
#dw_df['j'] = dw_df['j'].astype(float)
#dw_df['T'] = dw_df['T'].astype(float)
#dw_df.i = dw_df.i.round(ndecimals)
#dw_df.j = dw_df.j.round(ndecimals)

#dw_df=dw_df[(dw_df['conf']=='Cnf-11050000')&(dw_df['i']>0.20663)&(dw_df['i']<0.20664)].sort_values(by='i')
#with pd.option_context('display.float_format', '{:0.10f}'.format):
#    print(dw_df)
#sys.exit()
#        (0.062, 'Cnf-11050000', 0.2066353491, 0.206638039, '0.00572412295231') was classified as non-dw


# split this task between parallel workers
elements_per_worker=20
elements_per_worker=20999
chunks=[list_neb_qs[i:i + elements_per_worker] for i in range(0, len(list_neb_qs), elements_per_worker)]
n_chunks = len(chunks)
print('We are going to submit {} chunks for the non dw \n'.format(n_chunks))



def process_chunk(chunk):
    worker_df=pd.DataFrame()
    # I search for the given configuration
    for element in chunk:
        T,conf,i,j,qs = element
        # was this element used for training?
        a = dw_df[(dw_df['T']==T)&(dw_df['conf']==conf)&(dw_df['i'].between(i-rounding_error,i+rounding_error))&(dw_df['j'].between(j-rounding_error,j+rounding_error))]
        if len(a)>1:
            print('Error! multiple correspondences in dw')
            sys.exit()
        elif len(a)==1:
            a['quantum_splitting']=qs
            worker_df = pd.concat([worker_df,a])
        else:
            print('Error: we do not have {}'.format(element))
          #  # check if this pair is a non-dw
          #  a = nondw_df[(nondw_df['T']==T)&(nondw_df['conf']==conf)&(nondw_df['i'].between(i-rounding_error,i+rounding_error))&(nondw_df['j'].between(j-rounding_error,j+rounding_error))]
          #  if len(a)>1:
          #      print('Error! multiple correspondences in non-dw')
          #      sys.exit()
          #  elif len(a)==1:
          #      print('{} was classified as non-dw'.format(element))
          #      print(nondw_df[(nondw_df['T']==T)&(nondw_df['conf']==conf)&(nondw_df['i'].between(i-rounding_error,i+rounding_error))&(nondw_df['j'].between(j-rounding_error,j+rounding_error))])
          #      print(dw_df[(dw_df['T']==T)&(dw_df['conf']==conf)&(dw_df['i'].between(i-rounding_error,i+rounding_error))&(dw_df['j'].between(j-rounding_error,j+rounding_error))])
          #      print(dw_df[(dw_df['T']==T)&(dw_df['conf']==conf)&(dw_df['i'].between(i-rounding_error,i+rounding_error))])
          #      print(dw_df[(dw_df['T']==T)&(dw_df['conf']==conf)])
          #      sys.exit()
          #      a['quantum_splitting']=qs
          #      worker_df = pd.concat([worker_df,a])
          #  else:
          #      print('Error: we do not have {}'.format(element))
          #     # print(dw_df[(dw_df['T']==T)&(dw_df['conf']==conf)&(dw_df['i'].between(i-rounding_error,i+rounding_error))&(dw_df['j'].between(j-rounding_error,j+rounding_error))])
          #      sys.exit()
    return worker_df
        
# Initialize the pool
pool = mp.Pool(mp.cpu_count())
# *** RUN THE PARALLEL FUNCTION
results = pool.map(process_chunk, [chunk for chunk in chunks] )
pool.close()
# and add all the new df to the final one
qs_df=pd.DataFrame()
for df_chunk in results:
    qs_df= pd.concat([qs_df,df_chunk])
print('Constructed the database of {} pairs'.format(len(qs_df)))

print(qs_df)
sys.exit()


# *******
# add the pretrained data (if any)
qs_df = pd.concat([qs_df,pretrain_df])
qs_df = qs_df.drop_duplicates()

print(qs_df)
sys.exit()

# This is the new training df that will be stored at the end 
new_training_df = qs_df
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

# Pick 10percent of this pairs for validation
N = len(qs_df)
Nval = int(0.1*N)
Ntrain = N -Nval
# shuffle
qs_df=qs_df.sample(frac=1, random_state=20, ignore_index=True)
# and slice
training_set = qs_df.iloc[:Ntrain]
validation_set = qs_df.iloc[Ntrain:N]

print('\nFrom the overall %d data we prepare:\n\t- training set of %d\n\t- validation set of %d\n\n'%(len(new_training_df),len(training_set),len(validation_set) ),flush=True)


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
training_set = training_set.drop(columns=['i','j','conf'])
# * Convert to float to have optimal performances!
training_set = TabularDataset(training_set).astype(float)

# train
# * I am excluding KNN because it is problematic
predictor = TabularPredictor(label='qs', path=model_path, eval_metric='accuracy').fit(training_set, time_limit=time_limit,  presets=presets,excluded_model_types=['KNN'])


# store
new_training_df.to_pickle('MLmodel/data-used-by-qspredictor-M{}.pickle'.format(M))
training_set.to_pickle('MLmodel/qs-prediction-training-set-M{}.pickle'.format(M))
validation_set.to_pickle('MLmodel/qs-prediction-validation-set-M{}.pickle'.format(M))
