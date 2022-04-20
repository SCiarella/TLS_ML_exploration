# TLS landscape exploration

### A machine learning package by Simone Ciarella
[**Installation**](#Installation)
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






The idea of this gitrepo is :
	- to have a directory containing all the glassy configurations, and  where a single ML model can predict the qs for all the pairs  
	- keep in the same directory the NEB calculations that will be automatically used by the ML model to increse its predictive power 

_______________________________________________________________________________________________________________


To achieve those goals a rigorous organization of the data are required.

In particular all the minima for different glasses and configurations will be organized as:
	Configurations/minima/T{T_i}/Cnf-{xxx}/{energy_i}.conf.txt

Then the NEB calculations used to train the ML model have to respect this structure: 
	NEB_calculations/T{T_i}/NON-DW.txt   		(list of non DW)
	NEB_calculations/T{T_i}/Qs_calcuations.txt 	(list of calculated Qs)

The ML model which is contained in the MLmodel directory will be able to do two things:
	[a] predict the qs for all the pairs generated from Configurations/* 
	[b] retrain itself using the data in NEB_calculations/*

Its output will be the following:
	output_ML/T{T_i}/predictedQs_T{T_i}.csv 	(list of predicted Qs)
	output_ML/T{T_i}/nonDW_T{T_i}.csv 	(list of non DW)

Notice that in the future the Configurations/ and MLmodel/ directories will become local in order to contain the size of the repo

_______________________________________________________________________________________________________________



So the general idea is to do the following:
(1) collect new data  
(2) run [a] 
(3) for the best X pairs of (2) run the NEB
(4) (optional) run [b]
(5) either go to (1) or (3)  

