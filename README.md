<img src="./doc/fig_highlight.png" width="1100" />

# Finding defects in glasses through machine learning

<img src="./doc/fig_main.png" width="1100" />


*by **Simone Ciarella**, Dmytro Khomenko, Ludovic Berthier, Felix C. Mocanu, David R. Reichman, Camille Scalliet and Francesco Zamponi*
  
Paper links: [**Nat Commun 14, 4229 (2023)**](https://www.nature.com/articles/s41467-023-39948-7)  [*(arXiv)*](https://arxiv.org/abs/2212.05582)

Structural defects control the kinetic, thermodynamic and mechanical properties of glasses. For instance, rare quantum tunneling two-level systems (TLS) govern the physics of glasses at very low temperatures. Because of their extremely low density, it is very hard to directly identify them in computer simulations. We introduce a machine learning approach to efficiently explore the potential energy landscape of glass models and identify desired classes of defects. We focus in particular on TLS and we design an algorithm that is able to rapidly predict the quantum splitting between any two amorphous configurations produced by classical simulations. This in turn allows us to shift the computational effort towards the collection and identification of a larger number of TLS, rather than the useless characterization of non-tunneling defects which are much more abundant. Finally, we interpret our machine learning model to understand how TLS are identified and characterized, thus giving direct physical insight into their microscopic nature. 

---
*In this repository, I share the code used to produce the main findings of the paper. I also show step by step how this approach can be*  **applied to any other** *state-to-state transitions problem.*

---
  

## State-to-state transitions with machine learning

[**Installation**](#installation)
| [**Quick run**](#quick-run)
| [**Reproduce TLS results**](#reproduce-tls-results)



The idea of this project is to use machine learning to **speed up** the exploration of glassy landscapes, that fundamentally govern the behavior of disordered materials and many other systems characterized by slow dynamics. This repository puts particular emphasis on the concept of *iterative learning* applied to *state-to-state* transitions.

State-to-state transitions refer to the phenomenon where a physical system undergoes a change from one quantized state to another, often involving processes like excitation or relaxation, such as in two-level systems.
State-to-state transitions are extremely interesting, but when the dynamics is slow they are also extremely hard to find. And it is even worse for glassy systems, because they are characterized by an exponentially large number of states.
The practical problem is that usually the time-evolution trajectory of the system does not explore directly most of the state-to-state transitions available, during the limited observation time.

The ML model that I propose in this repository is able to investigate all the possible pairs of states (even the ones that the trajectory has not crossed yet) and rapidly (<img src="https://latex.codecogs.com/svg.image?10^{-5}" /> s) predicts crucial properties for the specific transition. This allows you to rapidly estimate if the pair is one of your desired state-to-state transition, that you can then collect and study. Overall the use of this ML approach **significantly reduces the computational load**, making such a research possible. 



## Installation

To install all the prerequired packages from a fresh conda environment run the following commands:
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

If you are interested in performing [SHAP](https://github.com/slundberg/shap) analysis of the trained model using the scripts provided here, you also need to install the following packages:
```
pip install shap seaborn
```

Then you can proceed with the download of this repository
```
cd ~
git clone https://github.com/SCiarella/TLS_ML_exploration.git
```

This package is *ready to run* and it just needs your input data. 

---
## Overview

<img src="./doc/fig_steps.png" width="1100" />

The repository consists a series of Python codes named `step[1-4].py`, that when executed in succession, perform **one step of the iterative procedure** to study state-to-state transitions. 

In summary, each of them accomplished the following task:
* *(not included)* **step0**:  data collection and preprocessing
* [**step1.py**](#step-1-training-the-classifier):  *(Filtering)* [re-]train the double well (DW) classifier
* [**step2.py**](#step-2:-classifier):  DW classification
* [**step3.py**](#step-3:-training-the-predictor):  *(Prediction)* [re-]train the predictor
* [**step4.py**](#step-4:-predicting-the-target-feature):  prediction of the target property of all the pairs (i.e. the quantum splitting)
* [*End of iteration step* $i$](#end-of-the-iteration-step:-new-calculations)


Those codes run using the content of the `MLmodel/` directory.
There is also a supporting file named `myparams.py` that allows the user to control the procedure as explained in detail in the next section.
Let's *discuss step by step* this procedure, using as example the TLS identification problem.


#### Step 0: Data collection and preprocessing

The first step of the procedure consists in collecting the relevant input features for the different pairs of states.
In this example, I use one of the collections of IS pairs that I discussed in the [paper](https://www.nature.com/articles/s41467-023-39948-7), which is stored on Zenodo at [TLS_input_data_Ciarella_et_al_2023](https://zenodo.org/record/8026630).
The dataframes contain pairs of configurations already preprocessed in order to have the following structure:
 
|              |feature 1| feature 2| ... | feature $N_f$ |
|--------------|---------|----------|-----|---------------|
|pair $i_1 j_1$|         |          |     |               |
|pair $i_2 j_1$|         |          |     |               |
|...           |         |          |     |               |
|pair $i_N j_N$|         |          |     |               |

Notice that the dataframes do not contain the output feature (i.e. the quantum splitting in the example), because I do not know its value for all the pairs and the goal of the whole procedure is in fact to calculate it, but only for the pairs which are more likely to be the *interesting* ones.

In a more general situation, the user must implement a `step0` procedure, to preprocess the raw data (i.e. XYZ configurations), into a dataframe containing the relevant information for each pair.
In the paper, I discuss how we ended up with our final set of features and the exclusion process that I used to save memory and space. 
In general, any number of features can be evaluated in `step0` and their specific importance strongly depends on the particular details of the problem. I discuss [here](https://www.nature.com/articles/s41467-023-39948-7) which features to use for questions similar to TLS and excitations. 
Overall, the user must identify the set of features that are better suited to capture the specific phenomenon of interest.
On top of the features that I discussed in the paper, useful additions could be [SOAP descriptors](https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html) or [bond-orientational order parameters](https://pyscal.org/en/latest/examples/03_steinhardt_parameters.html).


#### Step 1: Training the classifier

Next, let's train the classifier. The role of the classifier is to exclude pairs that are evidently not in the target group. In our example of TLS search, we know that a non-DW pair can not form a TLS, so we separate them a priori. 
In addition to the input file containing all the features, step-1 makes use of a pretraining set of size $K_0^c$ for the iterative training specified as `myparams.pretraining_classifier`. The pretraining set has to be placed in the `MLmodel/` directory. I provide an example for the TLS problem.

The pretraining dataframe contains the following information:

|              |feature 1|  ... | feature $N_f$ | is in class to keep ? |
|--------------|---------|------|---------------|:---------------------:|
|pair $i_1 j_1$|         |      |               |           {0,1}       |
|pair $i_2 j_1$|         |      |               |           {0,1}       |
|...           |         |      |               |           ...         |
|pair $i_N j_N$|         |      |               |           {0,1}       |

where the additional binary variable is set to $1$ if the pair is a good candidate for the target search (i.e. a DW), and $0$ if it is not a good candidate and we would like to discard it.
This will be the base for the initial training. 
It is likely that training the model a single time will already achieve good performance if $K_0^c$ is large enough (around $10^4$ pairs for the DW) and the sample is representative.
Notice that it is also **possible** to train the model **without pretraing**, but then the first few steps of the iterative procedure will probably show poor performance, and play the role of the pretraining.
The uses has total freedom in designing the approach that is more suited to the specific situation.

If it is not the first time that you are running *step_1.py* because you are at step $i>0$ of the iterative scheme, then the program needs to include in its training set the new pairs that you have studied after step-4 (at the end of the iterative step). This must be done by specifying in `myparams.calculations_classifier` the name of the file that lists the results from the calculations performed at the previous iterative step, following the suggestion of `step_4.py`. This file must be placed in the directory `exact_calculations/In_file_label/`, where the subdirectory In_file_label corresponds to `myparams.In_file` without its extension `.*`. 


#### Step 2: Classifier

The next step is to apply the classifier trained during step-1 to the full collection of pairs, in order to identify the good subgroup that can contain interesting pairs. 
To do so, the user can simply run `step2.py`. This will produce as output `output_ML/{In_file_label}/classified_{In_file_label}.csv` which is the dataframe containing the information of all the pairs classified in class-1, which are then promising. Later, steps 3 and 4 will perform their operations **only on this subset** of good pairs.



#### Step 3: Training the predictor

We can now train the predictor to ultimately estimate the target feature. As a reminder, the target feature was the quantum splitting or the energy barrier in the context of the TLS search discussed in the reference paper. 
In addition to the file generated by `step2.py` that contains all the pairs estimated to be in the interesting class, step-3 makes use of a pretraining set of size $K_0$ for the iterative training.
The name of this dataframe has to be specified as `myparams.pretraining_predictor`, and it must be placed in the `MLmodel/` directory.
The pretraining file contains the following information:

|              |feature 1|  ... | feature $N_f$ | target_feature |
|--------------|---------|------|---------------|:--------------:|
|pair $i_1 j_1$|         |      |               |                |
|pair $i_2 j_1$|         |      |               |                |
|...           |         |      |               |                |
|pair $i_N j_N$|         |      |               |                |

This will be the base for the initial training. Notice that it is also possible to train the model a single time and already achieve good performance if $K_0$ is large enough (around $10^4$ pairs for the TLS) and the sample is representative.

If this is not the first time that you are running *step_3.py* because you are at step $i>0$ of the iterative scheme, then the program needs to include in its training set the new pairs that you have studied following step-4 (at the end of the iterative step).
This must be done by specifying in `myparams.calculations_predictor` the name of the file that lists the results from the exact calculations over the pairs that have been suggested during the previous step of iterative training. This file must be placed in the directory `exact_calculations/In_file_label/`, where the subdirectory In_file_label corresponds to `myparams.In_file` without its extension `.*`. 



#### Step 4: Predicting the target feature

The final step of one iteration is to use the ML model to predict the target feature. Running `step4.py` will perform this prediction, and produce as output two files.
The first one is:
```
output_ML/{In_file_label}/predicted_{In_file_label}_allpairs.csv 	
```
which contains the prediction of `target_feature` for all the pairs available in `myparams.In_file`.

The second output file is:
```
output_ML/{In_file_label}/predicted_{In_file_label}_newpairs.csv 	
```
Similarly to the other file it reports the predicted `target_feature`, but it includes **only the new pairs** for which the exact value of `target_feature` is not already known from the exact calculations. This is *fundamental* because the iterative training procedure has to pick the next $K_i$ candidates from this restricted list, in order to avoid repetitions.


#### End of the iteration step: New calculations

Finally, at the end of each iteration step the user has to perform the analysis of `target_feature` for a new set of pairs, following the indications of `output_ML/{In_file_label}/predicted_{In_file_label}_newpairs.csv`.  
Noticeably, this operation will not be time consuming since the role of the ML model is to identify *good* pairs to analyze, while excluding most of them. The user has to select the number $K_i$ of new pairs to analyze each step, according to the specific details of the problem.    
As a reference, for the [TLS](https://www.nature.com/articles/s41467-023-39948-7) I measure `target_feature` (the quantum splitting) only for the $K=500$ top (new) pairs, according to `step_4.py`. Notice that this number $K$ is $< 0.00001\%$ of the total number of pairs. And the ML approach is so precise that we have been able to identify an unexpectedly large number of TLS, that would have been an insurmountable obstacle for classical non-ML methods. 

After performing the new measurements the iteration step is officially concluded, and the user can go back to step-1. It is also possible (and even suggested) to collect new data in the meantime. In this case the next iteration will restart from step-0, because the new data needs to be processed into the feature dataframe.
The users can perform as many iterations as they want. I suggest to keep track of the percentage of positive findings over time, and stop iterating when this rate gets too low. In the TLS reference, I performed $\mathcal{O}(10)$ iterations.

Refer to the [TLS paper](https://www.nature.com/articles/s41467-023-39948-7) and the [**Quick run**](#quick-run) section to get more details.


#### myparams.py

The supporting file `myparams.py` allows the user to set the desired parameters and hyperparameters. Here is reported the list with all the parameters that can be set in this way:
* **In_file**: name of the input file
* **pretraining_classifier**: name of the pretraining file for the classifier
* **pretraining_predictor**: name of the pretraining file for the predictor
* **calculations_classifier**: name of the file containing the list of pairs calculated in class-0
* **calculations_predictor**: name of the file containing the calculation of the target feature
* **class_train_hours**: training time in hours for the classifier
* **pred_train_hours**: training time in hours for the predictor
* **Fast_class**: if True use a lighter ML model for classification, with worse performance but better inference time 
* **Fast_pred**: if True use a lighter ML model for prediction, with worse performance but better inference time
* **ij_decimals**: number of significant digits to identify the states. If they are labeled using an integer number you can set this to 0
* **validation_split**: ratio of data that go into the validation set


#### Test the model

Finally, I also provide two test codes to evaluate the results of the ML model:
* `test1.py` will compare the predicted target feature with its exact value, over the validation set that was not used to train the model
* `test2.py` will perform the [SHAP](https://github.com/slundberg/shap) analysis for the trained model

The output of both tests will be stored in `output_ML/{In_file_label}/`.
  

---
## Quick run

The first step is to correctly set the parameters in `myparams.py` in order to point to the correct location for the input files.  
The most fundamental and necessary file is the dataframe containing all the available pairs `In_data/{myparams.In_file}`. 
Then in order to start the iterative procedure some initial observations are required. These can either be pretraining sets in `MLmodel/{myparams.pretraining_classifier}` and `MLmodel/{myparams.pretraining_predictor}` or alternatively some calculations of pairs contained in `In_data/{myparams.In_file}` that must be stored in `exact_calculations/{In_file_label}/{myparams.calculations_classifier}` and `exact_calculations/{In_file_label}/{myparams.calculations_classifier}`.

After this, it is possible to run the first full iteration consisting of `step[1-4].py`.
Finally, this will produce the output file `output_ML/predicted_{In_file_label}_newpairs.csv` that contains the predicted `target_feature` for all the available pairs:

| conf | i | j | target_feature |
|:----:|:-:|:-:|:--------------:|
| ...  |...|...|...             |
|      |   |   |                |

the dataframe contains only the pairs for which the exact calculation is not available and it is sorted based on the value of `target_feature`.

The final step of the iteration consists in calculating the exact value of `target_feature` for the best $K_i$ pairs, which corresponds to the first $K_i$ lines of `output_ML/predicted_{In_file_label}_newpairs.csv` if the target is a low value of `target_feature`.
You can reiterate this procedure as many times as you want and add new input pairs at any iteration.  
In the [paper](https://www.nature.com/articles/s41467-023-39948-7), I discuss some criteria to decide the value of $K_0$, the number of iterations and the stopping criterion.


---
## Reproduce TLS results

In order to reproduce the TLS result, I have provided the following [**Zenodo** directory](https://zenodo.org/record/8026630). 
This directory contains the dataframes `QS_T{0062,007,0092}.feather` corresponding to the pairs that have been used for the TLS analysis with all their relevant features, at the three temperatures.
In combination with the `exact_calculations/*/*.csv` containing the results of the NEB calculations (provided in this repository), the TLS data can be processed using the pipeline discussed above to reproduce all the findings reported in the [paper](https://www.nature.com/articles/s41467-023-39948-7).

Finally, in the directory `TLS_pairs_Khomenko_et_al_2020` I report the data corresponding to the smaller set of TLS pairs identified in *Khomenko et al. PRL 124.22 (2020): 225901*.