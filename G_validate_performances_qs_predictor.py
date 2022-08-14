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
import time
import multiprocess as mp
import matplotlib.pyplot as plt
import matplotlib
import myparams


# This code validates the performances of the qs predictor

if __name__ == "__main__":
    M = myparams.M
    T = myparams.T
    Tlabel = str(T).replace('.','')
    print('\n*** Requested to validate the qs predictor at T={} (M={})'.format(T,M))
    ndecimals=10
    rounding_error=10**(-1*(ndecimals+1))
    thresh_tls= 0.00151
    
    save_path='MLmodel/qs-regression-M{}-T{}'.format(M,Tlabel)
    validation_set = pd.read_feather('MLmodel/qs-prediction-validation-set-M{}-T{}.feather'.format(M,Tlabel))
    training_set = pd.read_feather('MLmodel/qs-prediction-training-set-M{}-T{}.feather'.format(M,Tlabel))
    
    # ********* RESULTS OVER THE TRAINING SET
    training_set = training_set.sort_values('quantum_splitting',ascending=False)
    qsMAX=2.5
    print('(Excluding from the plot pairs with qs>{})'.format(qsMAX))
    training_set=training_set[training_set['quantum_splitting']<qsMAX]
    # * Convert to float to have optimal performances!
    training_set_nolab = TabularDataset(training_set.drop(columns=['i','j','conf','quantum_splitting','10tominusquantum_splitting'])).astype(float)
    y_true_val = np.power(10, -training_set['10tominusquantum_splitting'])  # values to predict
    
    
    # Load the model
    predictor = TabularPredictor.load(save_path) 
    predictor.persist_models()
    # predict
    y_pred_by_AI = predictor.predict(training_set_nolab)
    y_pred_by_AI = np.power(10, -y_pred_by_AI)
    

    print('\n\nPerformances over the training set:')
    perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
    print(perf)
    print('\n')
    
    
    # Make the prediction plot: qs_real vs qs_train 
    fig, axs = plt.subplots()
    x = y_true_val
    y = y_pred_by_AI
    # draw a reference line
    axs.plot([min(x),max(x)], [min(x),max(x)],'b--', alpha=1, lw=1)
    hb = axs.hexbin(x,y,cmap='summer',mincnt=1,gridsize=75, xscale='log', yscale='log', norm=matplotlib.colors.LogNorm())
    axs.set_ylabel('quantum splitting (AI)', size=15)
    axs.set_xlabel('quantum splitting (True)', size=15)
    plt.yscale('log') 
    plt.xscale('log') 
    axs.legend()
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_ML/qs-true_vs_AI_trainingset.png',dpi=150)
    plt.close()
    
    
    
    
    # ********* RESULTS OVER THE TEST SET
    # * Convert to float to have optimal performances!
    validation_set= validation_set.sort_values('quantum_splitting',ascending=False)
    validation_set_nolab = validation_set.drop(columns=['i','j','conf','quantum_splitting','10tominusquantum_splitting'])
    validation_set_nolab = TabularDataset(validation_set_nolab).astype(float)
    y_true_val = np.power(10, -validation_set['10tominusquantum_splitting'])  # values to predict
    
    # predict
    y_pred_by_AI = predictor.predict(validation_set_nolab)
    y_pred_by_AI = np.power(10, -y_pred_by_AI)
    
    print('\n\nPerformances over the test set:')
    perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
    print(perf)

    
    # Make the prediction plot: qs_real vs qs_train 
    fig, axs = plt.subplots()
    x = y_true_val
    y = y_pred_by_AI
    y = np.fabs(y_pred_by_AI) ##########
    # draw a reference line
    axs.plot([min(x),max(x)], [min(x),max(x)],'b--', alpha=1, lw=1)
    hb = axs.hexbin(x,y,cmap='summer',mincnt=1,gridsize=75, xscale='log', yscale='log', norm=matplotlib.colors.LogNorm())
    axs.set_ylabel('quantum splitting (AI)', size=15)
    axs.set_xlabel('quantum splitting (True)', size=15)
    plt.yscale('log') 
    plt.xscale('log') 
    axs.legend()
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_ML/qs-true_vs_AI_testset.png',dpi=150)
    plt.close()
    
    
    # Plot the efficiency by measuring how many of the NEB that we actually run were tls
    # then load the info about all the pairs
    pairs_df = pd.read_feather('MLmodel/input_features_all_pairs_M{}-T{}.feather'.format(M,Tlabel))
    
    def process_chunk(chunk):
        worker_df=pd.DataFrame()
        # I search for the given configuration
        for element in chunk:
            T,conf,i,j,qs = element
            a = pairs_df[(pairs_df['T']==T)&(pairs_df['conf']==conf)&(pairs_df['i'].between(i-rounding_error,i+rounding_error))&(pairs_df['j'].between(j-rounding_error,j+rounding_error))]
            if len(a)>1:
                print('Error! multiple correspondences in dw')
                sys.exit()
            elif len(a)==1:
                b = a.copy()
                b['quantum_splitting']=qs
                worker_df = pd.concat([worker_df,b])
    #        else:
    #            print('Error: we do not have {}'.format(element))
    #            sys.exit()
        return worker_df
    
    print('\n\n\n')
    Tdir='./NEB_calculations/T{}'.format(T)
    print('\nCalculating efficiency at T={}'.format(T))
    list_neb_qs=[]
    with open('{}/Qs_calculations.txt'.format(Tdir)) as qs_file:
        lines = qs_file.readlines()
        for line in lines:
            conf = line.split()[0]
            i,j = line.split()[1].split('_')
            i = round(float(i),ndecimals)
            j = round(float(j),ndecimals)
            qs = line.split()[2]
            list_neb_qs.append((T,conf,i,j,qs))
    
    # split this task between parallel workers
    elements_per_worker=10
    chunks=[list_neb_qs[i:i + elements_per_worker] for i in range(0, len(list_neb_qs), elements_per_worker)]
    n_chunks = len(chunks)
    print('We are going to submit {} chunks to get the data\n'.format(n_chunks))
    
    # Initialize the pool
    pool = mp.Pool(mp.cpu_count())
    # *** RUN THE PARALLEL FUNCTION
    results = pool.map(process_chunk, [chunk for chunk in chunks] )
    pool.close()
    # and add all the new df to the final one
    qs_df=pd.DataFrame()
    missed_dw=0
    for df_chunk in results:
        qs_df= pd.concat([qs_df,df_chunk])
    if len(qs_df)<1:
        print('Error: for none of our available data we have NEB calculation, so it is not possible to validate our model prediction.')
        sys.exit()
    print('Constructed the database of {} pairs.\nPredicting'.format(len(qs_df)))
    
    y_pred_by_AI = predictor.predict(qs_df.drop(columns='quantum_splitting'))
    qs_df['quantum_splitting_PREDICTED'] = np.power(10, -y_pred_by_AI)
    qs_df=qs_df.sort_values(by='quantum_splitting_PREDICTED',ascending=True)
    
    x = []
    y = []
    tls = 0
    neb = 0
    qs_df['quantum_splitting']= qs_df['quantum_splitting'].astype(float)
    print('Iterating to find TLS')
    for index, row in qs_df.iterrows():
        neb +=1
        if float(row['quantum_splitting'])<thresh_tls:
            tls +=1
        x.append(neb)
        y.append(tls)
    
    fig, axs = plt.subplots()
    axs.plot(x,y)
    axs.set_ylabel('# TLS', size=15)
    axs.set_xlabel('# NEBs', size=15)
    plt.xscale('log') 
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_ML/T{}/TLS-search-efficiency.png'.format(T),dpi=150)
    plt.close()
    
    true=len(qs_df[(qs_df['quantum_splitting']<thresh_tls)&(qs_df['quantum_splitting_PREDICTED']<thresh_tls)])
    false_pos=len(qs_df[(qs_df['quantum_splitting']>=thresh_tls)&(qs_df['quantum_splitting_PREDICTED']<thresh_tls)])
    false_neg=len(qs_df[(qs_df['quantum_splitting']<thresh_tls)&(qs_df['quantum_splitting_PREDICTED']>=thresh_tls)])
    false=len(qs_df[(qs_df['quantum_splitting']>=thresh_tls)&(qs_df['quantum_splitting_PREDICTED']>=thresh_tls)])
    
    # Plot with confusion matrix
    qs_df = qs_df[(qs_df['quantum_splitting']<1) & (qs_df['quantum_splitting_PREDICTED']<1)]
    fig, axs = plt.subplots()
    x = qs_df['quantum_splitting']
    y = qs_df['quantum_splitting_PREDICTED']
    #
    axs.plot([min(x),max(x)], [min(x),max(x)],'k', alpha=1, lw=0.4)
    hb = axs.hexbin(x,y,cmap='summer',mincnt=1,gridsize=75, xscale='log', yscale='log', norm=matplotlib.colors.LogNorm())
    axs.set_ylabel('quantum splitting (AI)', size=15)
    axs.set_xlabel('quantum splitting (True)', size=15)
    ymin,ymax=axs.get_ylim()
    xmin,xmax=axs.get_xlim()
    tls_df=qs_df[qs_df['quantum_splitting']<thresh_tls]
    ntls=len(tls_df)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    for y_tls_dens_counter in [thresh_tls, 2*thresh_tls, 5*thresh_tls]:
        print('Running nebs for qs_predicted<{}, we get {}/{} of the TLS'.format(y_tls_dens_counter,len(tls_df[tls_df['quantum_splitting_PREDICTED']<y_tls_dens_counter]),ntls)) 
        dens_tls = round(float(len(tls_df[tls_df['quantum_splitting_PREDICTED']<y_tls_dens_counter])/ntls)*100,2) 
        nnebs = len(qs_df[qs_df['quantum_splitting_PREDICTED']<y_tls_dens_counter])
        axs.plot([xmin,xmax], [y_tls_dens_counter,y_tls_dens_counter],'k--', label='_nolegend_', alpha=1, lw=1, zorder=10)
        plt.text(xmax, y_tls_dens_counter, r'$\downarrow$ {}% of TLS, from {} NEB'.format(dens_tls,nnebs), fontsize=6, color='black', va='center', ha='right',zorder=100,bbox=props)
    axs.fill_between([xmin,thresh_tls], [thresh_tls,thresh_tls], facecolor='palegreen', label='True [n={}]'.format(true),zorder=-100, interpolate=True)
    axs.fill_between([xmin,thresh_tls], [ymax,ymax], [thresh_tls,thresh_tls], facecolor='paleturquoise', label='False negative [n={}]'.format(false_neg),zorder=-100, interpolate=True)
    axs.fill_between([thresh_tls,xmax], [thresh_tls,thresh_tls], facecolor='palegoldenrod', label='False positive [n={}]'.format(false_pos),zorder=-100, interpolate=True)
    axs.fill_between([thresh_tls,xmax], [ymax,ymax], [thresh_tls,thresh_tls], facecolor='lightcoral', label='Negative [n={}]'.format(false),zorder=-100, interpolate=True)
    plt.yscale('log') 
    plt.xscale('log') 
    axs.legend()
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_ML/T{}/confusion-matrix.png'.format(T),dpi=150)
    plt.close()
    
    qs_df.reset_index().to_feather('output_ML/T{}/neb_qs_df.feather'.format(T), compression='zstd')
    
        
