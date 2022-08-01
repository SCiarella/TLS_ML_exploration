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



if __name__ == "__main__":
    M = myparams.M
    ndecimals=10
    rounding_error=10**(-1*(ndecimals+1))

    Tdir='NEB_calculations/T0.062'
    T=float(Tdir.split('/T')[-1])

    qs_df=pd.read_pickle('output_ML/T{}/neb_qs_df.pickle'.format(T))
    print(qs_df)
    old_data_df=pd.read_csv('./new-TLS-from-old-database-T0062.csv')
    old_data_df=old_data_df[old_data_df['quantum_splitting']>0]
    old_data_df=old_data_df[old_data_df['quantum_splitting']<0.01]
    print(old_data_df.sort_values('quantum_splitting',ascending=False))
 

    fig, axs = plt.subplots(1,2,sharey=True)
    plt.subplots_adjust(wspace=0)



    thresh_tls= 0.00151
    # ********************************************
    # Right Panel :  new glasses
    qs_df = qs_df[(qs_df['quantum_splitting']<1) & (qs_df['quantum_splitting_PREDICTED']<1)]
    true=len(qs_df[(qs_df['quantum_splitting']<thresh_tls)&(qs_df['quantum_splitting_PREDICTED']<thresh_tls)])
    false_pos=len(qs_df[(qs_df['quantum_splitting']>=thresh_tls)&(qs_df['quantum_splitting_PREDICTED']<thresh_tls)])
    false_neg=len(qs_df[(qs_df['quantum_splitting']<thresh_tls)&(qs_df['quantum_splitting_PREDICTED']>=thresh_tls)])
    false=len(qs_df[(qs_df['quantum_splitting']>=thresh_tls)&(qs_df['quantum_splitting_PREDICTED']>=thresh_tls)])
    x = qs_df['quantum_splitting']
    y = qs_df['quantum_splitting_PREDICTED']
    min_overall = min(min(x),min(y))
    max_overall = max(max(x),max(y))
    axs[1].plot([min_overall,max_overall],[min_overall,max_overall], 'k', alpha=1, lw=0.4)
    ymin,ymax=axs[1].get_ylim()
    xmin,xmax=axs[1].get_xlim()
    axs[0].set_xlim(min_overall,max_overall)
    axs[0].set_ylim(min_overall,max_overall)
    axs[1].set_xlim(min_overall,max_overall)
    axs[1].set_ylim(min_overall,max_overall)
    hb = axs[1].hexbin(x,y,cmap='summer',mincnt=1,gridsize=75, xscale='log', yscale='log', norm=matplotlib.colors.LogNorm())
    tls_df=qs_df[qs_df['quantum_splitting']<thresh_tls]
    ntls=len(tls_df)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    for y_tls_dens_counter in [thresh_tls, 2*thresh_tls, 5*thresh_tls]:
        print('Running nebs for qs_predicted<{}, we get {}/{} of the TLS'.format(y_tls_dens_counter,len(tls_df[tls_df['quantum_splitting_PREDICTED']<y_tls_dens_counter]),ntls)) 
        dens_tls = round(float(len(tls_df[tls_df['quantum_splitting_PREDICTED']<y_tls_dens_counter])/ntls)*100,2) 
        nnebs = len(qs_df[qs_df['quantum_splitting_PREDICTED']<y_tls_dens_counter])
        axs[1].plot([xmin,xmax], [y_tls_dens_counter,y_tls_dens_counter],'k--', label='_nolegend_', alpha=1, lw=1, zorder=10)
        axs[1].text(xmax, y_tls_dens_counter, r'$\downarrow$ {}% of TLS, from {} NEB'.format(dens_tls,nnebs), fontsize=6, color='black', va='center', ha='right',zorder=100,bbox=props)
    axs[1].fill_between([xmin,thresh_tls], [thresh_tls,thresh_tls], facecolor='palegreen', label='True [n={}]'.format(true),zorder=-100, interpolate=True)
    axs[1].fill_between([xmin,thresh_tls], [ymax,ymax], [thresh_tls,thresh_tls], facecolor='paleturquoise', label='False negative [n={}]'.format(false_neg),zorder=-100, interpolate=True)
    axs[1].fill_between([thresh_tls,xmax], [thresh_tls,thresh_tls], facecolor='palegoldenrod', label='False positive [n={}]'.format(false_pos),zorder=-100, interpolate=True)
    axs[1].fill_between([thresh_tls,xmax], [ymax,ymax], [thresh_tls,thresh_tls], facecolor='lightcoral', label='Negative [n={}]'.format(false),zorder=-100, interpolate=True)
    axs[1].set_yscale('log') 
    axs[1].set_xscale('log') 
    axs[1].legend(fontsize=8)
    #axs[1].set_title(label=r'ML-driven TLS search at $T=0.062$')
    axs[1].set_title(label=r'New data ($T=0.062$)')

    #******************8
    # *** Left Panel :  old glasses new pairs
    thresh_tls= 0.0017
    old_data_df = old_data_df[(old_data_df['quantum_splitting']<1) & (old_data_df['quantum_splitting_PREDICTED']<1)]
    true=len(old_data_df[(old_data_df['quantum_splitting']<thresh_tls)&(old_data_df['quantum_splitting_PREDICTED']<thresh_tls)])
    false_pos=len(old_data_df[(old_data_df['quantum_splitting']>=thresh_tls)&(old_data_df['quantum_splitting_PREDICTED']<thresh_tls)])
    false_neg=len(old_data_df[(old_data_df['quantum_splitting']<thresh_tls)&(old_data_df['quantum_splitting_PREDICTED']>=thresh_tls)])
    false=len(old_data_df[(old_data_df['quantum_splitting']>=thresh_tls)&(old_data_df['quantum_splitting_PREDICTED']>=thresh_tls)])
    x = old_data_df['quantum_splitting']
    y = old_data_df['quantum_splitting_PREDICTED']
    axs[0].plot([xmin,xmax], [xmin,xmax],'k', alpha=1, lw=0.4)
    hb = axs[0].hexbin(x,y,cmap='summer',mincnt=1,gridsize=50, xscale='log', yscale='log', norm=matplotlib.colors.LogNorm())
    ymin,ymax=axs[0].get_ylim()
    xmin,xmax=axs[0].get_xlim()
    axs[0].fill_between([xmin,thresh_tls], [thresh_tls,thresh_tls], facecolor='palegreen', label='True [n={}]'.format(true),zorder=-100, interpolate=True)
    axs[0].fill_between([xmin,thresh_tls], [ymax,ymax], [thresh_tls,thresh_tls], facecolor='paleturquoise', label='False negative [n={}]'.format(false_neg),zorder=-100, interpolate=True)
    axs[0].fill_between([thresh_tls,xmax], [thresh_tls,thresh_tls], facecolor='palegoldenrod', label='False positive [n={}]'.format(false_pos),zorder=-100, interpolate=True)
    axs[0].fill_between([thresh_tls,xmax], [ymax,ymax], [thresh_tls,thresh_tls], facecolor='lightcoral', label='Negative [n={}]'.format(false),zorder=-100, interpolate=True)
    axs[0].set_yscale('log') 
    axs[0].set_xscale('log') 
    new_tls=old_data_df[old_data_df['quantum_splitting']<thresh_tls+0.0001]
    n_new_tls = len(new_tls)
    x = new_tls['quantum_splitting']
    y = new_tls['quantum_splitting_PREDICTED']
    axs[0].scatter(x,y,  c='blue', edgecolor='white', label='New TLS [n={}]'.format(n_new_tls)+' (excluded in Ref.XXX)',zorder=100)
    #axs[0].scatter(x,y,  c='blue', edgecolor='white', label='New TLS [n={}]'.format(n_new_tls) +r'{\fontsize{50pt}{3em}\selectfont{} (excluded in Ref.XXX)}',zorder=100)


    # *** FINAL options
    #cb = fig.colorbar(hb, ax=axs)
    #cb.set_label('counts')
    axs[0].set_ylabel('quantum splitting (AI)', size=12)
    axs[0].set_xlabel('quantum splitting (True)', size=12)
    axs[1].set_xlabel('quantum splitting (True)', size=12)
    axs[0].legend(fontsize=8)
    axs[0].set_title(label=r'Old data ($T=0.062$)')
    plt.tight_layout()
    plt.savefig('output_ML/fig4.png',dpi=350)
    plt.close()
    
