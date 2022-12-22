from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import sklearn
import shap
import gzip
import io
import os
import sys
import fnmatch
import pickle
import time
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import myparams


import warnings
warnings.filterwarnings('ignore')


class AutogluonWrapper_p:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        # the wrapper also transforms back the output using this transformation that we imposed for the training    
        return np.power(10,-self.ag_model.predict(X))

class AutogluonWrapper_c:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)

In_label = myparams.In_file.split('/')[-1].split('.')[0]
print('\n*** SHAP analysis for {}'.format(In_label))

try:
    shapdir='./output_ML/{}'.format(In_label)
    os.makedirs(shapdir)
except OSError:
    pass



data_predictor = pd.read_feather('./MLmodel/predictor-training-set-{}.feather'.format(In_label))
data_classifier = pd.read_feather('./MLmodel/classifier-training-set-{}.feather'.format(In_label))

# shuffle the dataset
data_predictor = data_predictor.sample(frac=1)
data_classifier = data_classifier.sample(frac=1)

# store some particular configs to later explain better
N_configs_to_explain_indetail = 4
specific_samples_classifier = data_classifier.iloc[:N_configs_to_explain_indetail]
specific_samples_predictor = data_predictor.iloc[:N_configs_to_explain_indetail]
X_specific_samples_p = specific_samples_predictor.drop(columns=['i','j','conf','target_feature'])
Y_specific_samples_p = specific_samples_predictor['target_feature']
X_specific_samples_c = specific_samples_classifier.drop(columns=['i','j','conf','class'])
Y_specific_samples_c = specific_samples_classifier['class']


# Separate input from output
X_p = data_predictor.drop(columns=['i','j','conf','target_feature'])
Y_p = data_predictor['target_feature']
X_c = data_classifier.drop(columns=['i','j','conf','class'])
Y_c = data_classifier['class']
list_of_features_p = list(X_p.columns)
list_of_features_c = list(X_c.columns)


# Load the predictor
model_path='MLmodel/prediction-{}'.format(In_label)
predictor = TabularPredictor.load(model_path, verbosity=1) 
predictor.persist_models()
# Load the classifier 
model_path='MLmodel/classification-{}'.format(In_label)
classifier = TabularPredictor.load(model_path, verbosity=1) 
classifier.persist_models()


# Create the AG wrappers
ag_wrapper_p = AutogluonWrapper_p(predictor, X_p.columns)
ag_wrapper_c = AutogluonWrapper_c(classifier, X_c.columns)


# Here I decide how many samples to use for the baseline
N_baseline = 3000 
X_p_for_bl = X_p.iloc[0:N_baseline,:]
X_c_for_bl = X_c.iloc[0:N_baseline,:]
# and then how many for the plots
N_SHAP = 5
X_p_for_shap = X_p.iloc[N_baseline:(N_baseline+N_SHAP),:]
X_c_for_shap = X_c.iloc[N_baseline:(N_baseline+N_SHAP),:]



# ***** I calculate the SHAP parameters for the predictor
print('\n**** Calculating the shap parameters for the predictor using {} samples as baseline and {} test points'.format(N_baseline,N_SHAP))
explainer_p = shap.Explainer(ag_wrapper_p.predict, X_p_for_bl)
shap_values_p = explainer_p(X_p_for_shap)
shap_values_specific_samples_p = explainer_p(X_specific_samples_p)


# *** Feature correlation plot
fig, myax = plt.subplots(figsize=(18,18))
# create dataframe of shap
shap_df = (pd.DataFrame(shap_values_p.data, columns=X_p.columns))
corr=shap_df.corr()
kot = corr[corr>=.5]
mask = np.zeros_like(kot)
mask[np.tril_indices_from(mask)] = True
sns.heatmap(kot,mask=mask, cmap='Greens')
plt.tight_layout()
plt.savefig('%s/predictor-correlation_features.png'%(shapdir),dpi=150)
plt.close()



# *** Waterfall plot
# it tells you why a single prediction of QS was made
# I do it only for certain samples you selected before
for i,shap_i in enumerate(shap_values_specific_samples_p):
    fig, myax = plt.subplots()
    shap.plots.waterfall(shap_i, max_display=7,show=False)
    plt.tight_layout()
    plt.savefig('%s/predictor-waterfall_targetfeat%g.png'%(shapdir,Y_specific_samples_p.iloc[i]),dpi=150)
    plt.close()


# *** Beeswarm plot
# it is a summary of the shap values for the most important features
fig, myax = plt.subplots()
shap.plots.beeswarm(shap_values_p, max_display=7, color=pl.get_cmap('RdYlGn_r'), log_scale=False,show=False)
plt.tight_layout()
plt.savefig('%s/predictor-beeswarm.png'%(shapdir),dpi=150)
plt.close()


# *** Partial dependence plots
# they tell you the expected target_feature given the value of a single input 
plt.switch_backend('agg')
for feature in list_of_features_p:
    print(' Partial dependence of %s'%feature)
    fig, myax = plt.subplots()
    myax.set_xscale('log')
    shap.plots.partial_dependence(
            feature, ag_wrapper_p.predict, X_p_for_shap, ice=False, hist=False,
        model_expected_value=False, feature_expected_value=False, ax=myax, show=False
    )
    myax.set_xscale('log')
    myax.set_ylabel('E[target | %s]'%feature)
    plt.tight_layout()
    plt.savefig('%s/predictor-partial_dependence_%s.png'%(shapdir,feature),dpi=150)
    plt.close()


# ***** I calculate the SHAP parameters for the classifier 
print('\n**** Calculating the shap parameters for the classifier using {} samples as baseline and {} test points'.format(N_baseline,N_SHAP))
explainer_c = shap.Explainer(ag_wrapper_c.predict, X_c_for_bl)
shap_values_c = explainer_c(X_c_for_shap)
shap_values_specific_samples_c = explainer_c(X_specific_samples_c)


# *** Feature correlation plot
fig, myax = plt.subplots(figsize=(18,18))
# create dataframe of shap
shap_df = (pd.DataFrame(shap_values_c.data, columns=X_c.columns))
corr=shap_df.corr()
kot = corr[corr>=.5]
mask = np.zeros_like(kot)
mask[np.tril_indices_from(mask)] = True
sns.heatmap(kot,mask=mask, cmap='Greens')
plt.tight_layout()
plt.savefig('%s/classifier-correlation_features.png'%(shapdir),dpi=150)
plt.close()



# *** Waterfall plot
# it tells you why a single prediction of QS was made
# I do it only for certain samples you selected before
for i,shap_i in enumerate(shap_values_specific_samples_c):
    fig, myax = plt.subplots()
    shap.plots.waterfall(shap_i, max_display=7,show=False)
    plt.tight_layout()
    plt.savefig('%s/classifier-waterfall_sample%d.png'%(shapdir,i),dpi=150)
    plt.close()


# *** Beeswarm plot
# it is a summary of the shap values for the most important features
fig, myax = plt.subplots()
shap.plots.beeswarm(shap_values_c, max_display=7, color=pl.get_cmap('RdYlGn_r'), log_scale=False,show=False)
plt.tight_layout()
plt.savefig('%s/classifier-beeswarm.png'%(shapdir),dpi=150)
plt.close()


# *** Partial dependence plots
# they tell you the expected target_feature given the value of a single input 
plt.switch_backend('agg')
for feature in list_of_features_c:
    print(' Partial dependence of %s'%feature)
    fig, myax = plt.subplots()
    myax.set_xscale('log')
    shap.plots.partial_dependence(
            feature, ag_wrapper_c.predict, X_c_for_shap, ice=False, hist=False,
        model_expected_value=False, feature_expected_value=False, ax=myax, show=False
    )
    myax.set_xscale('log')
    myax.set_ylabel('E[class | %s]'%feature)
    plt.tight_layout()
    plt.savefig('%s/classifier-partial_dependence_%s.png'%(shapdir,feature),dpi=150)
    plt.close()
