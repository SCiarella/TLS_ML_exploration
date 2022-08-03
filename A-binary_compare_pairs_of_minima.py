import numpy as np
import math
import gzip
import io
import os
import sys
import fnmatch
import csv
import glob
from shutil import copyfile
import pandas as pd
import myparams
#import multiprocessing as mp
import multiprocess as mp
from scipy.io import FortranFile
import time



# The goal of this code is to prepare a file for each T*/Conf*ij pair 
# Each file will be a df where each row consists in   
#   DE + Dxyz * M particles

# you have to rerun this code each time you add new minima for a glass, or you add new glasses

def dist(i,j,Lhalf,L):
    #distance between coordinate i and j using PBC
    distance = i - j
    while distance > Lhalf:
        distance -= L
    while distance < -Lhalf:
        distance += L
    return distance

def aver(i,j,Lhalf):
    # average position between coordinate i and j
    # not trivial due to PBC
    ijaver = j + dist(i,j,Lhalf,2*Lhalf)*0.5
    while ijaver > Lhalf:
        ijaver -= 2*Lhalf
    while ijaver < -Lhalf:
        ijaver += 2*Lhalf
    return ijaver

def displacement(ix,iy,iz,jx,jy,jz,Lhalf):
    L = 2*Lhalf
    dx = dist(ix,jx,Lhalf,L)
    dy = dist(iy,jy,Lhalf,L)
    dz = dist(iz,jz,Lhalf,L)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def displacement_v2(ix,iy,iz,jx,jy,jz,Lhalf):
    L = 2*Lhalf
    dx = dist(ix,jx,Lhalf,L)
    dy = dist(iy,jy,Lhalf,L)
    dz = dist(iz,jz,Lhalf,L)
    return dx*dx + dy*dy + dz*dz

#def displacement_v2(ix,iy,iz,jx,jy,jz,Lhalf):
#    L = 2*Lhalf
#    dx = dist(ix,jx,Lhalf,L)
#    dy = dist(iy,jy,Lhalf,L)
#    dz = dist(iz,jz,Lhalf,L)
#    if math.fabs(dx)+math.fabs(dy)+math.fabs(dz) < 0.01:
#        return 0
#    else:
#        return math.sqrt(dx*dx + dy*dy + dz*dz)

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

    if not os.path.exists('.temp'):
        os.makedirs('.temp', exist_ok=True)
    
    
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
            pairs_per_worker=100
            chunks=[difflist[i:i + pairs_per_worker] for i in range(0, len(difflist), pairs_per_worker)]
            n_chunks = len(chunks)
            print('We are going to submit {} chunks\n'.format(n_chunks))
    
    
            # This function prepares the panda database of the pairs that are in this chunk
            def process_chunk(chunk):
                worker_df=pd.DataFrame()
                for pair in chunk:
                    i, j = pair

                    start_time=time.time()

                    # the first info I need is DE
                    DE = float(j)-float(i)

                    # If DE is larger than 1, then the pair will never be a dw nor a tls
                    if DE*1500 > 1:
                        continue

                    # check if either i or j have been already processed
                    cnf_name = confdir.split('/')[-1].split('_')[0]
                    ipreprocessed = '.temp/{}_{}.pickle'.format(cnf_name,i)
                    jpreprocessed = '.temp/{}_{}.pickle'.format(cnf_name,j)
                    # Check for i
                    if os.path.isfile(ipreprocessed):
                        i_df = pd.read_pickle(ipreprocessed)
                    else:
                        # If Not I read it from the binary format
                        ifilename = '{}/{}.conf'.format(confdir,i)
                        try:
                            ifile = FortranFile(ifilename, 'r')
                        except Exception as e:
                            print('\nError: ')
                            print(e)
                            sys.exit(0)
                        idata = []
                        # Read N
                        [Ni]=ifile.read_ints(np.int32)
                        # Read L
                        ibox=ifile.read_reals(np.float64)
                        irxyz = []
                        for n in range(Ni):
                            irxyz.append( ifile.read_reals(np.float64) )
                        i_df = pd.DataFrame(irxyz, columns=['r','x','y','z'])
                        i_df['N']=Ni
                        # I am interested in half of L for PBC
                        i_df['L']=ibox[0]/2
                        # * Store
                        i_df.to_pickle(ipreprocessed)
                        ifile.close()
                    # Check for j
                    if os.path.isfile(jpreprocessed):
                        j_df = pd.read_pickle(jpreprocessed)
                    else:
                        jfilename = '{}/{}.conf'.format(confdir,j)
                        try:
                            jfile = FortranFile(jfilename, 'r')
                        except Exception as e:
                            print('\nError: ')
                            print(e)
                            sys.exit(0)
                        jdata = []
                        [Nj]=jfile.read_ints(np.int32)
                        jbox=jfile.read_reals(np.float64)
                        jrxyz = []
                        for n in range(Nj):
                            jrxyz.append( jfile.read_reals(np.float64) )
                        j_df = pd.DataFrame(jrxyz, columns=['r','x','y','z'])
                        j_df['N']=Nj
                        j_df['L']=jbox[0]/2
                        j_df.to_pickle(jpreprocessed)
                        jfile.close()

                
                    # check that N and L are consistent
                    if i_df['N'].iloc[0]!=j_df['N'].iloc[0]:
                        print('Error of i-j file energy inconsistence')
                        sys.exit()
                    N = i_df['N'][0]
                    if i_df['L'].iloc[0]!=j_df['L'].iloc[0]:
                        print('Error of i-j file boxsize inconsistence')
                        sys.exit()
                    L = i_df['L'][0]

                    

                    # make the dictionaty of all the differencies into a df
                    diff_df = pd.concat([i_df, j_df], axis=1)
                    diff_df = diff_df.set_axis(['ri','xi','yi','zi','Ni','Li','rj','xj','yj','zj','Nj','Lj'], axis=1, inplace=False)
                    diff_df['displacement'] = np.vectorize(displacement)(diff_df['xi'],diff_df['yi'],diff_df['zi'],diff_df['xj'],diff_df['yj'],diff_df['zj'], diff_df['Li'])

                    diff_df = diff_df.sort_values('displacement',ascending=False)
                    if diff_df['displacement'].iloc[0] <1e-10:
                        print('Warning: small displacement (largest is %f)'%diff_df['displacement'].iloc[0])
                        print(diff_df)
                        print(diff_df['displacement'])
                        sys.exit()


                    # calculate the total displacement and the PR 
                    diff_df['dpow2'] = np.square(diff_df['displacement'])
                    diff_df['dpow4'] = np.square(diff_df['dpow2'])
                    total_displacement = diff_df['displacement'].sum() 
                    sum_dpow2 = diff_df['dpow2'].sum() 
                    sum_dpow4 = diff_df['dpow4'].sum() 
                    PR = sum_dpow2*sum_dpow2/sum_dpow4

    
                    # ** I store only the M_to_store particles that displaced the most
                    diff_df = diff_df[:M]

                    # For those single pairs measure the average position of the particles that displaced the most
                    diff_df['x'] = np.vectorize(aver)(diff_df['xi'],diff_df['xj'], diff_df['Li'])
                    diff_df['y'] = np.vectorize(aver)(diff_df['yi'],diff_df['yj'], diff_df['Li'])
                    diff_df['z'] = np.vectorize(aver)(diff_df['zi'],diff_df['zj'], diff_df['Li'])

                    # then remove the columns that are not needed
                    single_pair_df=diff_df[['x','y','z','displacement']]
    
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

                    # and also add a column for the PR and displacements
                    single_pair_df['total_displacement']=total_displacement
                    single_pair_df['PR']=PR
    
                    # Finally append the line to the database
                    worker_df = pd.concat([worker_df,pd.DataFrame(single_pair_df).T])
            
                #print('Worker [{} / {}] done'.format(str(mp.current_process()).split('-')[1].split('\'')[0],n_chunks))
                return(worker_df)
            
            # Initialize the pool
            pool = mp.Pool(1)
            #pool = mp.Pool(mp.cpu_count())
            
            # *** RUN THE PARALLEL FUNCTION
            results = pool.map(process_chunk, [chunk for chunk in chunks] )
            
            # Step 3: Don't forget to close
            pool.close()
            
            # and add all the new df to the final one
            for df_chunk in results:
                full_df = pd.concat([full_df,df_chunk])
            
            print('\n*Done\n\n')
            full_df.to_pickle(full_df_name)

            # remove all the tempfiles
            cnf_name = confdir.split('/')[-1].split('_')[0]
            temp_list = glob.glob('.temp/{}_*.pickle'.format(cnf_name))
            for tempfile in temp_list:
                os.remove(tempfile)
