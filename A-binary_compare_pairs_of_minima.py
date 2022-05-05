import numpy as np
import math
import itertools
import random
import gzip
import gzip
import io
import os
import sys
import fnmatch
import csv
import glob
from shutil import copyfile
import numpy as np
import pandas as pd
import myparams
#import multiprocessing as mp
import multiprocess as mp
from scipy.io import FortranFile


# The goal of this code is to prepare a file for each T*/Conf*ij pair 
# Each file will be a df where each row consists in   
#   DE + Dxyz * M particles

# you have to rerun this code each time you add new minima for a glass, or you add new glasses

def dist(i,j,Lhalf):
    #distance between coordinate i and j using PBC
    distance = i - j
    while distance > Lhalf:
        distance -= 2*Lhalf
    while distance < -Lhalf:
        distance += 2*Lhalf
    return distance
def aver(i,j,Lhalf):
    # average position between coordinate i and j
    # not trivial due to PBC
    ijaver = j + dist(i,j,Lhalf)*0.5
    while ijaver > Lhalf:
        ijaver -= 2*Lhalf
    while ijaver < -Lhalf:
        ijaver += 2*Lhalf
    return ijaver
def displacement(ix,iy,iz,jx,jy,jz,Lhalf):
    return math.sqrt(dist(ix,jx,Lhalf)**2 +dist(iy,jy,Lhalf)**2 +dist(iz,jz,Lhalf)**2)

def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

if __name__ == "__main__":
    M = myparams.M
    
    pairs_dir='Configurations/pairs'
    if not os.path.exists(pairs_dir):
        os.makedirs(pairs_dir, exist_ok=True)
    
    
    # loop over all the temperatures separately
    list_T = glob.glob('Configurations/minima/*')
    for Tdir in list_T:
        T = Tdir.split('/T')[1].split('/')[0]
        print(Tdir)
        print('\n\n****** Processing T={}'.format(T))
    
        ensure_dir('output_ML/T{}/'.format(T))
        ensure_dir('NEB_calculations/T{}/'.format(T))
    
        # loop over all the glasses separately
        list_conf = glob.glob('{}/*'.format(Tdir))
        for confdir in list_conf: 
            conf = confdir.split('/')[3]
            print('conf={}\t\t(T={})'.format(conf,T))
    
            # Check if the full db exhist or it has to be created
            full_df_name='{}/pairs-T{}-{}-M{}.pickle'.format(pairs_dir,T,conf,M)
            if os.path.isfile(full_df_name):
                print('\n%s already exhist, so I import and add to it\n'%full_df_name)
                full_df=pd.read_pickle(full_df_name)
            else:
                print('\n%s does NOT exhist, so I create it\n'%full_df_name)
                full_df=pd.DataFrame()
    
            # Now I check which pairs I have available
            list_i = sorted([(x.split('/')[-1].split('.conf')[0]) for x in glob.glob('{}/*'.format(confdir))])
            list_ij = []
            for index, i in enumerate(list_i):
                for j in list_i[(index+1):]:
                    list_ij.append((i,j))
    
        
            # from which I only keep the one that are not already processed into the df
            processed_pairs = []
            len_df=len(full_df)
            if len_df>0:
                df_is = full_df['i'].tolist()
                df_js = full_df['j'].tolist()
                processed_pairs = [(x,y) for x,y in zip(df_is, df_js)]
                difflist = list(set(list_ij) - set(processed_pairs))
            else:
                difflist = list_ij
            
            print('\nFrom {} pairs we processed {} so we now have only {} to do'.format(len(list_ij), len(processed_pairs), len(difflist) ))
            if len(list_ij)!=len(processed_pairs)+len(difflist):
                print('ERROR: there is a consstency problem in the numbers above')
                sys.exit()
    
            
            # and I split them in chunks to give to separate parallel workers
            pairs_per_worker=200
            chunks=[difflist[i:i + pairs_per_worker] for i in range(0, len(difflist), pairs_per_worker)]
            n_chunks = len(chunks)
            print('We are going to submit {} chunks\n'.format(n_chunks))
    
    
            # This function prepares the panda database of the pairs that are in this chunk
            def process_chunk(chunk):
                worker_df=pd.DataFrame()
                for pair in chunk:
                    i, j = pair
                    ifilename = '{}/{}.conf'.format(confdir,i)
                    jfilename = '{}/{}.conf'.format(confdir,j)

                
                    # the first info I need is DE
                    DE = float(j)-float(i)

                    # If DE*N is larger than 1, then the pair will never be a dw nor a tls
                    if DE*1500 > 1:
                        continue
    
                    # then I read the two files looking for the M particles that displaced the most
                    # * notice that they are binary files
                    try:
                        ifile = FortranFile(ifilename, 'r')
                        jfile = FortranFile(jfilename, 'r')
                    except Exception as e:
                        print('\nError: ')
                        print(e)
                        sys.exit(0)
                    
                    [Ni]=ifile.read_ints(np.int32)
                    [Nj]=jfile.read_ints(np.int32)
                    if Ni!=Nj:
                        print('Error of i-j file energy inconsistence')
                        sys.exit()
                    N = Ni

                    # Now I read and compare 
                    ibox=ifile.read_reals(np.float64)
                    jbox=jfile.read_reals(np.float64)
                    # I am interested in half of L for PBC
                    if ibox[0]==jbox[0]:
                        L=ibox[0]/2
                    else:
                        print('Error of i-j file boxsize inconsistence')
                        sys.exit()
                    
                    # Now I read the coordinates and process the variables
                    pair_variables=[]
                    for n in range(N):
                        ir, ix, iy, iz = ifile.read_reals(np.float64)
                        jr, jx, jy, jz = jfile.read_reals(np.float64)
                        d = {'x': aver(ix,jx,L)}
                        d['y']=aver(iy,jy,L)
                        d['z']=aver(iz,jz,L)
                        d['displacement']=displacement(ix,iy,iz,jx,jy,jz,L)
                        pair_variables.append(d)

                    ifile.close()
                    jfile.close()
    
                    # make the dictionaty of all the differencies into a df
                    single_pair_df = pd.DataFrame(pair_variables).astype(float)
    
                    single_pair_df = single_pair_df.sort_values('displacement',ascending=False)
                    if single_pair_df['displacement'].iloc[0] <1e-10:
                        print('Warning: small displacement (largest is %f)'%single_pair_df['displacement'].iloc[0])
                        print(single_pair_df)
                        print(single_pair_df['displacement'])
                        sys.exit()
    
                    # To save space I store only the M_to_store particles that displaced the most
                    single_pair_df = single_pair_df[:M]
    
                    # make it into a single row
                    single_pair_df=single_pair_df.reset_index(drop=True)
                    single_pair_df = single_pair_df.stack()
                    single_pair_df.index = single_pair_df.index.map('{0[1]}_{0[0]}'.format)
    
                    # and add a column with the Delta_E
                    single_pair_df['Delta_E']=DE*1500
    
                    # plus the columns about i j T conf
                    # notice that you have to keep i and j as string in order to find their files
                    single_pair_df['i']=i
                    single_pair_df['j']=j
                    single_pair_df['T']=T
                    single_pair_df['conf']=conf
    
                    # Finally append the line to the database
                    worker_df = pd.concat([worker_df,pd.DataFrame(single_pair_df).T])
            
                #print('Worker [{} / {}] done'.format(str(mp.current_process()).split('-')[1].split('\'')[0],n_chunks))
                return(worker_df)
            
            # Initialize the pool
            pool = mp.Pool(mp.cpu_count())
            #pool = mp.Pool(Nprocessors)
            
            # *** RUN THE PARALLEL FUNCTION
            results = pool.map(process_chunk, [chunk for chunk in chunks] )
            
            # Step 3: Don't forget to close
            pool.close()
            
            # and add all the new df to the final one
            for df_chunk in results:
                full_df = pd.concat([full_df,df_chunk])
            
            print('\n*Done\n\n')
            full_df.to_pickle(full_df_name)
