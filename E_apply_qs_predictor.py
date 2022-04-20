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

# This code runs the qs prediction for all the pairs that have been predicted to be dw 


try:
    with open("M_val.txt") as f:
        M=int(f.readlines()[0].strip('\n'))
        print('M={}'.format(M))
except Exception as error:
    print('Error: {}'.format(Exception))
    sys.exit()

M = int(M)
natoms = 1500
save_path='MLmodel/qs-regression-M{}'.format(M)

# Load the model
predictor = TabularPredictor.load(save_path) 
predictor.persist_models()

# I loop over all the T separately
list_T = glob.glob('output_ML/T*')
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
    dw_df.to_csv('{}/predictedQs_T{}.csv'.format(Tdir,T))

    # plot the distribution
    plt.figure();
    ecdf = ECDF(dw_df["quantum_splitting"])
    Norm = len(dw_df) / (nglass*natoms)
    plt.loglog(ecdf.x, Norm * ecdf.y/ecdf.x, "o");
    n0 = Norm*np.quantile(ecdf.y/ecdf.x, 0.001)
    plt.axhline(n0, ls="--")
    plt.xlabel(r"$E$");
    plt.ylabel(r"$\dfrac{n(E)}{E}$");
    plt.savefig("{}/splitting_cdf_T{}.png".format(Tdir,T), dpi=150, bbox_inches="tight");

