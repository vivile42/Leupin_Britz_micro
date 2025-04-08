Code accompanying Leupin and Britz, Pre-stimulus microstates The momentary state of the brain and  the bodily signals y exert independently influence perceptual awareness at the discrimination threshold.
We here provide the code for preprocessing, epoching, and pre-stimulus analysis of EEG data. EEG data can be made available upon request.

#Code organization
Each folder is generally organized with a main, helper and constants script.The base folder contains some helper functions to filter through the data directories.

The main scripts contain the code that must be run.
The helper scripts contain the helper functions and classes used to run the code.
The constant files contain constants that are called in the script.

**Preprocessing**
Order to run preprocessing
1) markers/markers_main.py
2) epochs/epochs_main.py
3) ICA
4) evoked/autoreject_main.py
5) microstates/micro_main.py
**statistics**
microstates/prestate_LMM.rmd
microstates/micro_figure.ipynb
microstates/microstates_parametres.ipynb

##Description
markers: analyzes cardiac and respiratory signals and generates markers to classify each stimulus according to the behavioral response and the cardiac / respiratory phase.
epochs: segments the EEG into epochs before artifact rejection and computes ICA solutions.
ICA: jupyter notebook to be applied to each subject to manually select ICA components to be rejected.
evoked:
autoreject: used the Autoreject procedure to clean the epoched data after the ICA.
microstates: extract epochs, find best cluster solution to explain data, fit cluster map to the data and compute parametres 

##Analyses

microstates/microstates_parametres.ipynb Compute descriptive parametres
microstates/micro_figure.ipynb code to generate the main figure
microstates/prestate_LMM.rmd code to compute the analysis on R
