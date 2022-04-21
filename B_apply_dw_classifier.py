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
import multiprocessing as mp

# This code runs the dw vs non-dw classifier over all the pairs 


try:
    with open("M_val.txt") as f:
        M=int(f.readlines()[0].strip('\n'))
        print('M={}'.format(M))
except Exception as error:
    print('Error: {}'.format(Exception))
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
pool = mp.Pool(mp.cpu_count())
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
try:
    old_df = pd.read_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))
except:
    print('First time running this code')
    old_df=pd.DataFrame()

if len(new_df)<len(old_df):
    print('\n***Error: input_features* has lenght {} while I find only {} pairs. This is only possible if you have lost data!'.format(len(old_df),len(new_df)) )
    sys.exit()

# Convert the data to the correct types
print('\nConverting data types')
new_df['i'] = new_df['i'].astype(float)
new_df['j'] = new_df['j'].astype(float)
new_df['T'] = new_df['T'].astype(float)
new_df['Delta_E'] = new_df['Delta_E'].astype(float)
new_df['conf'] = new_df['conf'].astype(str)
old_df['i'] = old_df['i'].astype(float)
old_df['j'] = old_df['j'].astype(float)
old_df['T'] = old_df['T'].astype(float)
old_df['conf'] = old_df['conf'].astype(str)
old_df['Delta_E'] = old_df['Delta_E'].astype(float)
for mi in range(int(M)):
    new_df['displacement_{}'.format(mi)] = new_df['displacement_{}'.format(mi)].astype(float) 
    old_df['displacement_{}'.format(mi)] = old_df['displacement_{}'.format(mi)].astype(float) 

# check which data are shared and which one are new
print('\nCross-check and merging')
used_df = pd.merge(old_df, new_df) 
Nnew=len(new_df)-len(used_df)

print('\n\t@@@@ Overall we have {} pairs, of which {} are new from the last time you run this'.format(len(new_df),Nnew))

# * Then I store this df to avoid having to redo it 
new_df.to_pickle('MLmodel/input_features_all_pairs_M{}.pickle'.format(M))


# *************
# (3) apply the dw Filter 
start= time.time()
classifier_save_path = 'MLmodel/dw-classification-M{}'.format(M)
print('\nUsing the DW filter trained in {}'.format(classifier_save_path))

print('\n* Classifier loading',flush=True)
dwclassifier = TabularPredictor.load(classifier_save_path) 

print('\n* Classifier starts',flush=True)
new_df['is_dw'] = dwclassifier.predict(new_df)
timeclass=time.time() -start

filtered_dw = new_df[ new_df['is_dw']>0 ] 
filtered_non_dw = new_df[ new_df['is_dw']<1 ] 
print('From the {} pairs, only {} are classified as dw (in {} sec = {} sec per pair), so {} are non-dw'.format(len(new_df), len(filtered_dw), timeclass, timeclass/len(new_df), len(filtered_non_dw)))

filtered_df_name='output_ML/T{}/DW_T{}.csv'.format(T,T)
filtered_dw.to_csv(filtered_df_name, index=False)

filtered_non_dw_name='output_ML/T{}/nonDW_T{}.csv'.format(T,T)
filtered_non_dw.to_csv(filtered_non_dw_name, index=False)
