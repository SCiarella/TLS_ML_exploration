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

# This code runs the dw vs non-dw classifier over all the pairs 




if len(sys.argv) > 1:
    M = sys.argv[1]
else:
    print('Error because I have not received IN')
    sys.exit()

# The absolute position is not interesting, so I only report the distance from particle 0
# notice that you have to evalaute the distance using PBC !
Lhalf= 0.5 * 11.447142426
def PBC_dist(x):
    while x > Lhalf:
        x-=2*Lhalf
    while x < -Lhalf:
        x+=2*Lhalf
    return x
M = int(M)


# *************
# (1) Load the T and conf to get a list of all the data 
list_df_pairs=[]
list_T = glob.glob('Configurations/minima/*')
for Tdir in list_T:
    T = Tdir.split('/T')[1].split('/')[0]
    list_df_pairs.extend( glob.glob('Configurations/pairs/pairs-T{}-*-M{}.pickle'.format(T,M)))
print('\n* A total of {} glasses are availabe to process'.format(len(list_df_pairs)))

# and I split the list in chunks to give to separate parallel workers
pairs_per_worker=10
chunks=[list_df_pairs[i:i + pairs_per_worker] for i in range(0, len(list_df_pairs), pairs_per_worker)]
n_chunks = len(chunks)
print('We are going to submit {} chunks\n'.format(n_chunks))

# *************
# (2) parallel worker to read the pair and check if some data are new
def process_chunk(chunk):
    worker_df=pd.DataFrame()
    for glass_df_name in chunk:
        #print('reading {}'.format(glass_df_name))
        glass_df=pd.read_pickle(glass_df_name)
        try:
            for m in range(1,M):
                x = glass_df['x_%d'%m] - glass_df['x_0']
                y = glass_df['y_%d'%m] - glass_df['y_0']
                z = glass_df['z_%d'%m] - glass_df['z_0']
                # PBC
                x = x.apply(PBC_dist)
                y = y.apply(PBC_dist)
                z = z.apply(PBC_dist)
                glass_df['0to%d'%m] = np.sqrt( x*x + y*y + z*z ) 
                glass_df = glass_df.drop(columns=['x_%d'%m,'y_%d'%m,'z_%d'%m])
            glass_df = glass_df.drop(columns=['x_0','y_0','z_0'])
        except:
            print('* Warning: {} has length {}'.format(glass_df_name,len(glass_df)))
        # Finally append the line to the database
        worker_df = pd.concat([worker_df,glass_df])
        
    print('Worker [{} / {}] done'.format(str(mp.current_process()).split('-')[1].split('\'')[0],n_chunks))
    return(worker_df)

# Initialize the pool
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
#pool = mp.Pool(Nprocessors)

# *** RUN THE PARALLEL FUNCTION
results = pool.map(process_chunk, [chunk for chunk in chunks] )

# Step 3: Don't forget to close
pool.close()

# and add all the new df to the final one
new_df=pd.DataFrame()
for df_chunk in results:
    new_df = pd.concat([new_df,df_chunk])

print('\n*Done reading the glasses')

# load the df containing all the pairs that I found last time
full_df = pd.read_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))

if len(new_df)<len(full_df):
    print('\n***Error: input_features* has lenght {} while I find only {} pairs. This is only possible if you have lost data!'.format(len(full_df),len(new_df)) )

# check which data are shared and which one are new
used_df = pd.merge(full_df, new_df) 
Nnew=len(new_df)-len(used_df)

print('Overall we have {} pairs, of which {} are new from last time'.format(len(new_df),Nnew))

# * Then I store this df to avoid having to redo it 
full_df.to_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))


# *************
# (3) apply the dw Filter 
start= time.time()
classifier_save_path = 'MLmodel/dw-classification-M{}'.format(M)
print('\nUsing the DW filter trained in {}'.format(classifier_save_path))

print('\n* Classifier loading',flush=True)
dwclassifier = TabularPredictor.load(classifier_save_path, verbosity=5) 
print('\n* Classifier starts',flush=True)
new_pairs_isdw = full_df
#new_pairs_isdw.drop(columns=['i','j','T','conf'])
new_pairs_isdw['is_dw'] = dwclassifier.predict(full_df.drop(columns=['i','j','T','conf']))
timeclass=time.time() -start

filtered_dw = new_pairs_isdw[ new_pairs_isdw['is_dw']>0 ].drop(columns='is_dw') 
filtered_non_dw = new_pairs_isdw[ new_pairs_isdw['is_dw']<1 ].drop(columns='is_dw') 
print('From the {} pairs, only {} are classified as dw (in {} sec), so {} are non-dw'.format(len(new_pairs_isdw), len(filtered_dw), timeclass, len(filtered_non_dw)))

filtered_df_name='output_ML/T{}/DW_T{}.pickle'.format(T,T)
filtered_dw.to_pickle(filtered_df_name)

filtered_non_dw_name='output_ML/T{}/nonDW_T{}.pickle'.format(T,T)
filtered_non_dw.to_pickle(filtered_non_dw_name)
