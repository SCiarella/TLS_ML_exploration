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
    In_label = myparams.In_file.split('/')[-1].split('.')[0]
    print('\n*** Testing the predictor for {}'.format(In_label))
    # specify the threshold above which the target feature is not relevant (to exclude from the plot)
    maximum_value_tg_feature=1e-1
    
    model_path='MLmodel/prediction-{}'.format(In_label)
    validation_set = pd.read_feather('MLmodel/predictor-validation-set-{}.feather'.format(In_label))
    
    # ********* RESULTS OVER THE TEST SET
    # * Convert to float to have optimal performances!
    validation_set= validation_set.sort_values('target_feature',ascending=False)
    validation_set = validation_set[validation_set['target_feature']<maximum_value_tg_feature]
    validation_set_nolab = validation_set.drop(columns=['i','j','conf','target_feature','10tominus_tg'])
    validation_set_nolab = TabularDataset(validation_set_nolab)
    y_true_val = np.power(10, -validation_set['10tominus_tg'])  # values to predict
    
    # Load the model
    predictor = TabularPredictor.load(model_path) 
    predictor.persist_models()
    # predict
    y_pred_by_AI = predictor.predict(validation_set_nolab)
    y_pred_by_AI = np.power(10, -y_pred_by_AI)
    

    print('\n\nPerformances:')
    perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
    print(perf)
    print('\n')
    
    
    # Make the prediction plot
    fig, axs = plt.subplots()
    x = y_true_val
    y = y_pred_by_AI
    # draw a reference line
    axs.plot([min(x),max(x)], [min(x),max(x)],'b--', alpha=1, lw=1)
    hb = axs.hexbin(x,y,cmap='summer',mincnt=1,gridsize=75, xscale='log', yscale='log', norm=matplotlib.colors.LogNorm())
    axs.set_ylabel('Target feature (AI)', size=15)
    axs.set_xlabel('Target feature (True)', size=15)
    plt.yscale('log') 
    plt.xscale('log') 
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('counts')
    plt.tight_layout()
    plt.savefig('output_ML/{}/{}-predicted_vs_true.png'.format(In_label,In_label),dpi=150)
    plt.close()
    
