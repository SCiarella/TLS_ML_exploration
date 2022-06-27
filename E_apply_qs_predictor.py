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
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib
import myparams

# This code runs the qs prediction for all the pairs that have been predicted to be dw 


M = myparams.M
natoms = 1500
save_path='MLmodel/qs-regression-M{}'.format(M)
low_thresh_qs=0.00001
ndecimals=10

# check if the model is there
if not os.path.isdir(save_path):
    print('Error: I am looking for the ML model in {}, but I can not find it. If this is the first time, you have to run F_* before this'.format(save_path))
    sys.exit()

# Load the model
predictor = TabularPredictor.load(save_path) 
predictor.persist_models()

# I loop over all the T separately
list_T = glob.glob('output_ML/T*')
if len(list_T)==0:
    print('There are no data in output_ML. Did you run B ?')
for Tdir in list_T:
    T = Tdir.split('/T')[1].split('/')[0]

    # Read the data
    dw_df = pd.read_csv('{}/DW_T{}.csv'.format(Tdir,T))
    nglass = len(dw_df['conf'].drop_duplicates())
    print('\n***Reading the data at T={}\nWe found {} glasses and a total of {} pairs\n\nStarting predictions'.format(T,nglass, len(dw_df)))

    # predict the qs
    X = dw_df.drop(columns=['i','j','conf','is_dw'])
    X = TabularDataset(X).astype(float)
    y_pred_by_AI = predictor.predict(dw_df)
    y_pred_by_AI = np.power(10, -y_pred_by_AI)
    print('The qs has been predicted. Now storing results')

    # store the prediction
    dw_df['quantum_splitting']=y_pred_by_AI
    dw_df = dw_df.sort_values(by='quantum_splitting')
    dw_df[['conf','i','j','quantum_splitting']].to_csv('{}/predictedQs_T{}_allpairs.csv'.format(Tdir,T),index=False)


    # load the NEB data for the plot
    list_neb_qs=[]
    with open('NEB_calculations/T{}/Qs_calculations.txt'.format(T)) as qs_file:
        lines = qs_file.readlines()
        for line in lines:
            qs = line.split()[2]
            list_neb_qs.append(float(qs))
    neb_qs=pd.DataFrame({'quantum_splitting':list_neb_qs})

    # plot the distribution
    fig,axs = plt.subplots();
    ecdf = ECDF(dw_df[dw_df['quantum_splitting']>low_thresh_qs]["quantum_splitting"])
    Norm = len(dw_df) / (nglass*natoms)
    plt.loglog(ecdf.x, Norm * ecdf.y/ecdf.x, "o", label='AI prediction');
    n0 = Norm*np.quantile(ecdf.y/ecdf.x, 0.001)
    plt.axhline(n0, ls="--")
    ecdf = ECDF(neb_qs[neb_qs['quantum_splitting']>low_thresh_qs]["quantum_splitting"])
    Norm = len(neb_qs) / (nglass*natoms)
    plt.loglog(ecdf.x, Norm * ecdf.y/ecdf.x, "o", label='NEB calculation');
    n0 = Norm*np.quantile(ecdf.y/ecdf.x, 0.001)
    plt.axhline(n0, color='r')
    plt.xlabel(r"$E$");
    plt.ylabel(r"$\dfrac{n(E)}{E}$");
    axs.legend()
    plt.savefig("{}/splitting_cdf_T{}.png".format(Tdir,T), dpi=150, bbox_inches="tight");
    plt.close()


    # Now I exclude the pairs that have already been used
    dw_df = dw_df.drop(columns=['is_dw']).round(decimals=10)
    used_df = pd.read_pickle('MLmodel/data-used-by-qspredictor-M{}.pickle'.format(M))
    used_df = used_df[used_df['i']!='NotAvail']
    used_df = used_df.drop(columns=['quantum_splitting'])
    used_df['i'] = used_df['i'].astype(float)
    used_df['j'] = used_df['j'].astype(float)
    used_df = used_df.round(decimals=10)
    dw_df.reset_index(drop=True)
    used_df.reset_index(drop=True)
    used_df = used_df[['i','j','conf']]
    dw_df['index'] = dw_df.index
    remove_df = pd.merge(dw_df, used_df) 
    remove_df = remove_df.set_index('index')
    dw_df = dw_df.drop(columns='index')
    dw_df = dw_df[~dw_df.isin(remove_df)].dropna()
    print('\n*For {} pairs we already run the NEB and we know the exact qs, so we do not need to predict them. Now we remain with {} pairs'.format(len(remove_df),len(dw_df)))

    # then I also exclude the pairs that I know are NON-dw
    list_nondw=[]
    with open('NEB_calculations/T{}/NON-DW.txt'.format(T)) as qs_file:
        lines = qs_file.readlines()
        for line in lines:
            conf = line.split()[0]
            i,j = line.split()[1].split('_')
            i = round(float(i),ndecimals)
            j = round(float(j),ndecimals)
            list_nondw.append((conf,i,j))
    nondw=pd.DataFrame(list_nondw,columns=['conf','i','j'])
    dw_df.reset_index(drop=True)
    dw_df['index'] = dw_df.index
    remove_df = pd.merge(dw_df, nondw) 
    remove_df = remove_df.set_index('index')
    dw_df = dw_df.drop(columns='index')
    dw_df = dw_df[~dw_df.isin(remove_df)].dropna()
    print('\n*We know that {} pairs are non-dw (from NEB), so we do not need to predict them.\nWe then finish with {} new pairs'.format(len(remove_df),len(dw_df)))

    # Storing
    dw_df[['conf','i','j','quantum_splitting']].to_csv('{}/predictedQs_T{}_newpairs.csv'.format(Tdir,T),index=False)
