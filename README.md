# TLS landscape exploration

### A machine learning package by Simone Ciarella
[**Installation**](#Installation)
| [**Quick run**](#Quickrun)
| [**Paper**](https://arxiv.org/pdf/111.pdf)



The idea of this project is to use machine learning to speed up the exploration of the landscape of glassy materials.
In particular two-level systems (TLS) are extremely interesting, but hard to find in molecular dynamics (MD) simulations.
This program is able to construct all the pairs of inherent structures (IS or energy minima) combining the provided IS and predict in $10^{-5}$ their quantum splitting, thus suggesting which one are likely to be TLS.


# Installation

To install all the prerequired packages from a fresh conda environment run the following
```
conda create -n tls_exploration
conda activate tls_environment
git clone https://github.com/SCiarella/autogluon
cd autogluon && ./full_install.sh
```

Then you can proceed with the download of this package
```
cd
git clone https://github.com/SCiarella/TLS_ML_exploration.git
```

The package is already ready to run and it just needs your new data. 


## Overview

The repository consist in a series of python codes named `[A-G]\_\.py` . 

In brief, each of them has the following task:
* **A**\*.py:  constructs all the pairs of minima
* **B**\*.py:  classifies double wells (dw)
* **C**\*.py:  re-trains the dw classifier
* **D**\*.py:  validates the performances of the dw classifier
* **E**\*.py:  predicts the qs of all the pairs
* **F**\*.py:  re-trains the qs predictor
* **G**\*.py:  validates the performances of the qs predictor

Those codes run behind the scene using the content of the MLmodel directory.
The other directories present in the project are the following:

#### Configurations/*
This contains the IS obtained from the simulations. In order for the codes to work, its content has to respect the following structure:
```
Configurations/minima/T{T_i}/Cnf-{xxx}/{energy_i}.conf.txt
```

#### NEB_calculations/*
This directory contains the *exact* results of the NEB calculations. We need to collect them for two reasons: 
* to validate the AI predictions 
* to re-train the models
The content of the directory has to respect the following convention:
```
NEB_calculations/T{T_i}/NON-DW.txt   		#(list of non DW)
NEB_calculations/T{T_i}/Qs_calcuations.txt 	#(list of calculated Qs)
```

#### output_ML/*
This is the directory that contains the output of the ML. 
*We use to ML predictions to decide for which pairs we run NEB*. In particular this information is contained in 
```
output_ML/T{T_i}/predictedQs_T{T_i}.csv 	
```
which lists the predicted qs for all the pairs we have available.


# Quick run

### First execution
Right after the installation, and *collecting the data* and storing them as explained [here](#Overview) , you can run:
```
python A_*
```
to prepare all the pairs, followed by
```
python C_* 
```
to train the dw classifier for the first time.

Then in order to predict if your unknown pairs are dw or not you run
```
python B_* 
```

Then in order to train the qs predictor you run
```
python F_* 
```
and then you can run
```
python E_* 
```
in order to predict the quantum splitting


### Each time you add new data
Each time you add new data in Configurations/minima, you have to run

```
python A_* && python B_* && python E_* 
```
in order to obtain predictions also for this new data.

---
**_Optionally:_** if you have new NEB results you can *re-train the ML models*.
In particular you can retrain the dw classifier with
```
python A_* && python C_* 
```
and you can train the qs predictor with
```
python A_* && python E_* 
```
---

### Validation
It is possible to evaluate the performances of the ML models by running
```
python D_* 
```
to test the dw classifier, and
```
python G_* 
```
to test the qs predictor.

The content of `output\_ML/dw\_classifier\_performances\.txt` will provide an estimate of the accuracy of the classifier evaluated over its training set followed up by the same measures over a test set that the model has not used for its training. You can expect that this second measure will be the overall performance of the classifier.

The qs predictor can be evaluated by looking at the plots in `output\_ML/qs-true\_vs\_AI\_t\*set.png` that will show the quality of the prediction over the training set and the test set (not used for training).
Additionally it is possible to look at `output\_ML/T\*/splitting\_cdf\_T\*.png` which reports the cumulativie distribution of the energy splitting, comparing the NEBs to the ML predictions, and `output\_ML/T\*/TLS-search-efficiency.png` which reports the efficiency of the ML approach showing how many NEBs are required to find all the TLS available so far.


## Output of the ML model
The most interesting output of the ML model is the database in `**output\_ML/T\*/predictedQs\_T\*.csv**`
It contains the quantum splitting prediction for all the pairs of minima, ordered from smallest to largest.
This means that the **_next neb that you should run_** are for the pairs at the beginning of this list.