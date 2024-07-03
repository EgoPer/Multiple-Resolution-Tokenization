# Multiple-Resolution Tokenisation with an Application to Pricing
This repository contains the modules that form the MRT and the some of the utility code used for piping data into it.

Due to commercial sensitivity not everything can be shared. We provide the implementation of the model as it was done in experiments but omit some of the data pre-processing. We also can not publish the data. To reproduce the model on a new dataset after installing the requirements we suggest the following steps:

1. The dataset should be processed such that each row contains all variables at that time step for a series. For static variables the values should be repeated. There should be a variable defining the groups that are normalised together (InputTypes.ID), a variable defining the position in time so that the dataset can be split (InputTypes.TIME), and target variables (InputTypes.TARGET). There needs to be a sequence id defining all rows which belong to a given series. For each series there needs to be a variable defining the ordering of each series so that it can be sorted (in our example this is *per_series_index* as defined by the quantisation code in *pricing_utils.py* and the sorting is already performed by the quantisation). 
2. Add the file path to the *run_experiments_pricing.sh* file and any necessary preprocessing to *run_pricing.py*.
3. In *utils.py* add column definitions to a list detailing the core column definitions to the *ExperimentConfig* class. A column definition is defined by a tuple (string name of variable, data type of variable, input type of variable). Core column definitions have to include the ID, TIME, and TARGET InputTypes. There can be multiple TARGET variables as this defines the channels of the problem (the variables which we want to forecast). Add a dictionary entry with experiment name: column definition to *self.column_definition_map*. An example is provided with *pricing_coldef*.
4. In *run_pricing.py* add other variables in the same format as the column definitions which should be included in the model.
5. This code is designed to work with pricing data in which each series is uniquely defined by the combination of product id, store id, and due date. In *run_pricing.py* you should define the column/variable which contains this sequence id with *data_formatter.sequence_id*.
6. Adjust hyperparameters to fit your forecasting problem in *run_experiments_pricing.sh*.


## File Guide
Note that this code is experimental. I appologise for the inefficiencies and redundancies and hope that replicating it is not too burdensome.
- *run_experiment_pricing.sh* : The shell script for running an experiment, the inputs are explained in the argument parser in *run_pricing.py*.
- *run_pricing.py* : The python script to run an experiment. Loads and processes data, creates and trains model, logs results.
- *MRT.py* : Contains modules that are used to construct the MRT and the lighting module that is actually used in experiments.
- *utils.py* : Utility functions used for data formatting and piping. Contains DataTypes and InputTypes which are used to define different forms of data associated with a time series process. Some of the code was developed from the code for the [Temporal Fusion Transformer](https://github.com/google-research/google-research/tree/master/tft).
- *training_utils.py* : Utility functions related to training. Mainly useful for piping.
- *pricing_utils.py* : Some utility functions related to pricing data specifically, includes data loading.
- *layers/* : Contains a few modules utilised in the MRT, code mostly borrowed from elsewhere.
  - *utils.py* : Modules for positional encoding, convenient transpose modules,... Mostly unused, copied from PatchTST repo.
  - *RevIN.py* : Module for reversible instance normalisation, copied [from](https://github.com/ts-kim/RevIN).
  - *attention.py* : Code for multihead attention, separated as multiple transformer models were in development at some stage.
- *requirements.txt* : Package dependencies.
