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
import matplotlib.pyplot as plt
import matplotlib


# This code takes all the available data (results from the NEB) and if they are more than the data that we already used to train the model, we retrain it 


# We need the desired M for the target df
try:
    with open("M_val.txt") as f:
        M=int(f.readlines()[0].strip('\n'))
        print('M={}'.format(M))
except Exception as error:
    print('Error: {}'.format(Exception))
    sys.exit()

save_path='MLmodel/qs-regression-M{}'.format(M)
validation_set = pd.read_pickle('MLmodel/qs-prediction-validation-set-M{}.pickle'.format(M))
training_set = pd.read_pickle('MLmodel/qs-prediction-training-set-M{}.pickle'.format(M))

# ********* RESULTS OVER THE TRAINING SET
label= 'quantum_splitting'
training_set = training_set.sort_values(label,ascending=False)
training_set_nolab = training_set.drop(columns=['i','j','conf',label])
# * Convert to float to have optimal performances!
training_set_nolab = TabularDataset(training_set_nolab).astype(float)
y_true_val = np.power(10, -training_set[label])  # values to predict




# Load the model
predictor = TabularPredictor.load(save_path, verbosity=3) 
predictor.persist_models()
# predict
y_pred_by_AI = predictor.predict(training_set_nolab)
y_pred_by_AI = np.power(10, -y_pred_by_AI)

print('\n\nPerformances over the training set:')
perf = predictor.evaluate_predictions(y_true=y_true_val, y_pred=y_pred_by_AI, auxiliary_metrics=True)
print(perf)



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
plt.savefig('output_ML/qs-true_vs_AI_trainingset_M%s.png'%(M),dpi=150)
plt.close()




# ********* RESULTS OVER THE TEST SET
# * Convert to float to have optimal performances!
validation_set= validation_set.sort_values(label,ascending=False)
validation_set_nolab = validation_set.drop(columns=['i','j','conf',label])
validation_set_nolab = TabularDataset(validation_set_nolab).astype(float)
y_true_val = np.power(10, -validation_set[label])  # values to predict

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
plt.savefig('output_ML/qs-true_vs_AI_testset_M%s.png'%(M),dpi=150)
plt.close()

