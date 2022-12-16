<img src="./doc/fig_main.png" width="1100" />

# Finding two-level systems in glasses through machine learning

*by Simone Ciarella, Dmytro Khomenko, Ludovic Berthier, Felix C. Mocanu, David R. Reichman, Camille Scalliet and Francesco Zamponi*
  
Paper link: [**arXiv: 2212.05582**](https://arxiv.org/abs/2212.05582)

Two-level systems (TLS) are rare quantum tunneling defects which govern the physics of glasses at very low temperature. Because of their extremely low density, it is very hard to directly identify them in computer simulations of model glasses. We introduce a machine learning approach to efficiently explore the potential energy landscape of glass models and identify two-level tunneling defects. We design an algorithm that is able to rapidly predict the quantum splitting between any two amorphous configurations produced by classical simulations. This in turn allows us to shift the computational effort towards the collection and identification of a larger number of TLS, rather than the useless characterization of non-tunneling defects which are much more abundant. Finally, we interpret our machine learning model to understand how TLS are identified and characterized, thus giving physical insight into the features responsible for their presence.

---
*In this repository we share the code used to produce the main findings of the paper. We also show step by step how this approach can be generalized to study other state-to-state transitions.*

---
  

## State-to-state transitions with machine learning

[**Installation**](#Installation)
| [**Quick run**](#Quick-run)
| [**Reproduce TLS results**](https://arxiv.org/abs/2212.05582)



The idea of this project is to use machine learning to **speed up** the exploration of the landscape of glassy materials or slow dynamics, with particular focus on the *iterative training* scheme that we introduced.
State-to-state transitions like two-level systems are extremely interesting, but when the dynamics is slow they are very hard to find, and the situation is even worse for glassy systems, characterized by an exponential number of states.
The problem is that often the trajectory of the system does not explore directly the targetted state-to-state transitions during the limited observation time.
The ML model that we propose constructs all the pairs of states (even the one that the trajectory never crossed) and rapidly (<img src="https://latex.codecogs.com/svg.image?10^{-5}" /> s) predicts target crucial properties for the specific transition, thus estimating if the pair is one of the desired transition and if precise calculation is needed. Overall this significantly reduces the computational load. 



## Installation

To install all the prerequired packages from a fresh conda environment run the following
```
conda create -n tls_exploration -y python=3.9
conda activate tls_exploration
conda install -y -c conda-forge statsmodels
conda install -c conda-forge multiprocess -y
git clone https://github.com/SCiarella/autogluon
cd autogluon && ./full_install.sh
```

> **_NOTE_**:  If you are a MacOS user you need to manually install the correct version of LibOMP via:
```
# brew install wget
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew uninstall libomp
brew install libomp.rb
rm libomp.rb
```


Then you can proceed with the download of this package
```
cd ~
git clone https://github.com/SCiarella/TLS_ML_exploration.git
```

The package is already ready to run and it just needs your new data. 

---
## Overview

<img src="./doc/fig_steps.png" width="1100" />

The repository consist in a series of python codes named `step[0-4].py` . 

In brief, each of them has the following task:
* **step0**.py:  data collection and preprocessing
* **step1**.py:  [re-]train the double well (DW) classifier
* **step2**.py:  DW classification
* **step3**.py:  [re-]train the predictor
* **step4**.py:  prediction of the target property of all the pairs (i.e. the quantum splitting)


Those codes run using the content of the MLmodel directory.
There is also a supporting file named `myparams.py` that allows the user to control the procedure as explained in detail in the next section.
Let's discuss step by step this procedure, using as example the TLS identification problem.


#### Step 0: Data collection and preprocessing

The first step of the procedure consist in collecting the relevant input features for the different pairs of states.
In the example `step0.py` we load the database of IS pairs that we use in our [paper](https://arxiv.org/abs/2212.05582), which is uploaded on [Zenodo](https://zenodo.org/) [TBD] and contains the input features discussed in the paper.
The user can then specify the correct input file name as `myparams.In_file` .
The input database is expected to have the following structure:
 
|              |feature 1| feature 2| ... | feature $N_f$ |
|--------------|---------|----------|-----|---------------|
|pair $i_1 j_1$|         |          |     |               |
|pair $i_2 j_1$|         |          |     |               |
|...           |         |          |     |               |
|pair $i_N j_N$|         |          |     |               |

Notice that the database does not contain the output feature (i.e. the quantum splitting), because we do not know its value for all the pairs and the goal of this procedure is to calculate it only for a small selected groups of pairs.
For a different problem than the one we discuss, we suggest to start with the inclusion of additional descriptors such as [SOAP](https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html) or [bond orientational order parameters](https://pyscal.org/en/latest/examples/03_steinhardt_parameters.html).


#### Step 1: Training the classifier

Next we train the classifier. The role of the classifier is to exclude pairs that are evidently not in the target group. In our example of TLS search we know that a non-DW pair can not form a TLS, so we separate them a priori. 
In addition to the input file containing all the features, step 1 makes use of a pretraining set of size $K_0^c$ for the iterative training specified as `myparams.pretraining_classifier`, that has to be placed in the `MLmodel/` directory.
The pretraining file contains the following information:

|              |feature 1|  ... | feature $N_f$ | is in class to exclude ? |
|--------------|---------|------|---------------|:------------------------:|
|pair $i_1 j_1$|         |      |               |           {0,1}          |
|pair $i_2 j_1$|         |      |               |           {0,1}          |
|...           |         |      |               |           ...            |
|pair $i_N j_N$|         |      |               |           {0,1}          |

where the additional binary variable is set to $1$ if the pair is a good candidate for the target search (i.e. a DW), and $0$ if not.
This will be the base for the initial training. Notice that it is also possible to train the model a single time and already achieve good performance, if $K_0^c$ is large enough (around $10^4$ pairs for the DW) and the sample is representative.

Furthermore, if the process is at any $i>0$ reiteration of the iterative training scheme, then the program needs to include in its training set the new pairs that have been calculated during the iterative procedure. This can be done by specifying in `myparams.calculations_classifier` the name of the file that lists the results from the exact calculations over the pairs that have been suggested during the previous step of iterative training. This file has to be located in the directory `exact_calculations/In_file_label/`, where the subdirectory In_file_label corresponds to `myparams.In_file` without its extension `.*` . 


#### Step 2: Classifier

The following step is to apply the classifier to the full collection of pairs in order to identify the good subgroup that can contain interesting pairs. 
To do so, the user has simply to run `step2.py`. This will produce as output `output_ML/{In_file_label}/classified_{In_file_label}.csv` which is the database containing the information of all the pairs classified in class-1. Steps 3-4 will perform their operations only on this subset of pairs.



#### Step 3: Training the predictor

We can now train the predictor to estimate the target feature. This corresponds to the quantum splitting or the energy barrier in the context of our TLS search. 
In addition to the file generated by `step2.py` that contains all the pairs estimated to be in the interesting class, step 3 makes use of a pretraining set of size $K_0$ for the iterative training specified as `myparams.pretraining_predictor`, that has to be placed in the `MLmodel/` directory.
The pretraining file contains the following information:

|              |feature 1|  ... | feature $N_f$ | target_feature |
|--------------|---------|------|---------------|:--------------:|
|pair $i_1 j_1$|         |      |               |                |
|pair $i_2 j_1$|         |      |               |                |
|...           |         |      |               |                |
|pair $i_N j_N$|         |      |               |                |

This will be the base for the initial training. Notice that it is also possible to train the model a single time and already achieve good performance, if $K_0$ is large enough (around $10^4$ pairs for the TLS) and the sample is representative.

Furthermore, if the process is at any $i>0$ reiteration of the iterative training scheme, then the program needs to include in its training set the new pairs that have been calculated during the iterative procedure. This can be done by specifying in `myparams.calculations_predictor` the name of the file that lists the results from the exact calculations over the pairs that have been suggested during the previous step of iterative training. This file has to be located in the directory `exact_calculations/In_file_label/`, where the subdirectory In_file_label corresponds to `myparams.In_file` without its extension `.*` . 


#### Step 4: Predicting the target feature

The final step of the iteration is to predict the target feature. Running `step4.py` will perform this prediction, and produce as output two files:
```
output_ML/{In_file_label}/predicted_{In_file_label}_allpairs.csv 	
```
containing the prediction of `target_feature` for all the pairs available in `myparams.In_file`, and
```
output_ML/{In_file_label}/predicted_{In_file_label}_newpairs.csv 	
```
that reports the predicted `target_feature` only for the pairs for which the exact calculation is not done. This is useful because the iterative training procedure has to pick the next $K_i$ candidates from this restricted list, in order to avoid repetitions.


#### myparams.py

The supporting file `myparams.py` allows the user to set the correct hyperpameters. Here it is reported the list with all the parameters that can be set in this way:
* **In_file**: name of the input file
* **pretraining_classifier**: name of the pretraining file for the classifier
* **pretraining_predictor**: name of the pretraining file for the predictor
* **calculations_classifier**: name of the file containing the list of pairs calculated in class-0
* **calculations_predictor**: name of the file containing the calculation of the target feature
* *class_train_hours*: training time in hours for the classifier
* *pred_train_hours*: training time in hours for the predictor
* **Fast_class**: if True use a lighter ML model for classification, with worse performance but better inference time 
* **Fast_pred**: if True use a lighter ML model for prediction, with worse performance but better inference time
* **ij_decimals**: number of significant digits to identify the states. If they are labeled using an integer number you can set this to 0
* **validation_split**: ratio of data that go into the validation set


| :no_entry:   | [Work in progress] We are updating the package. The content below is not consistent with the present version of the repository|
|--------------|:------------------------------------------------------------------------------------------------------------------------------|


...
----




# Quick run

### First execution
Right after the installation, and *collecting the data* and storing them as explained [here](#Overview) , you can run:
```
python A_*
```
to prepare all the pairs, followed by
```
python B_* && python C_* 
```
to train the dw classifier for the first time.
| :memo:        | Notice that some data are already provided to train the dw classifier and the qs predictor       |
|---------------|:------------------------|

Then in order to predict if your unknown pairs are dw or not you run once again
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


| :point_up:        | If you have new NEB results you can **re-train the ML models**. |
|---------------|:------------------------|

In particular you can retrain the dw classifier with
```
python A_* && python C_* 
```
and you can train the qs predictor with
```
python A_* && python E_* 
```

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

The content of `output_ML/dw_classifier_performances.txt` will provide an estimate of the accuracy of the classifier evaluated over its training set followed up by the same measures over a test set that the model has not used for its training. You can expect that this second measure will be the overall performance of the classifier.

The qs predictor can be evaluated by looking at the plots in `output_ML/qs-true_vs_AI_t*set.png` that will show the quality of the prediction over the training set and the test set (not used for training).
Additionally it is possible to look at `output_ML/T*/splitting_cdf_T*.png` which reports the cumulativie distribution of the energy splitting, comparing the NEBs to the ML predictions, and `output_ML/T*/TLS-search-efficiency.png` which reports the efficiency of the ML approach showing how many NEBs are required to find all the TLS available so far.

---
## Output of the ML model
The most interesting output of the ML model is the database in `output_ML/T*/predictedQs_T*_newpairs.csv`
It contains the quantum splitting prediction for all the pairs of minima, ordered from smallest to largest.
This means that the **_next NEBs that you should run_** are for the pairs at the beginning of this list.
