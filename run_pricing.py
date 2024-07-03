import copy
from pathlib import Path, PurePosixPath
import warnings
import argparse
import csv
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import List
import os
from functools import reduce

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from utils import ExperimentConfig, CsvDictWriter, DataTypes, InputTypes
from MRT import LightningMultipleResolutionTokenization
from pricing_utils import determine_cutoffs, quantize_by_row, ForecastingDatasetPricing


report_file = "report.csv"
def parse_args():
    """Parse the arguments for an MRI experiment"""

    parser = argparse.ArgumentParser(
        description="Multiple Resolution Transformer experiment parameters"
    )

    parser.add_argument('--experiment', type=str, default="pricing", help='Name of dataset/experiment')
    parser.add_argument('--forecast_horizon', type=int, help='Forecast horizon')
    parser.add_argument('--time_features', type=int, default=1, choices = [0,1], help='Boolean coded as 0 or 1, include time features or not')
    parser.add_argument('--loss', type=str, default="RMSE", choices = ["MAE","RMSE","Smooth"], help='Loss function name')
    parser.add_argument('--d_model', type=int, default= 8, help='Latent dimension of the transformer')
    parser.add_argument('--d_ff', type=int, default= 8, help='Latent dimension of the fully connected layer in the encoder/decoder')
    parser.add_argument('--d_mixer', type=int, default= 8, help='Latent dimension of the mixer in the divider')
    parser.add_argument('--n_head', type=int, default= 1, help='Number of attention heads, must divide d_model')
    parser.add_argument('--N', type=int, default= 1, help='Number of stacked transformer blocks')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--ablation_options', nargs='+', default=["",""], help= 
                        '''
        'shared_weighs' : 'weights of all additional modules are shared across channel if true (default true, recommended true)',
        'auxiliary' : 'include auxiliary variables?',
        'channel_mixer' : 'add a channel mixer component which learns tokens across channels using a mixer architecture (only recommended if gains are expected from concurrent cross-series information',
        'squeeze_static' : 'transform number of static tokens to deepest partition boolean, not recommended when there are few static variables',
        'squeeze_known_temporal_variables' : 'transform number of known temporal auxiliary variable tokens or channel mixer tokens to deepest partition (recommended True)',
        'PE' : 'add learnable positional encoding',
                        '''
                       )
    parser.add_argument('--densest_partition', type=int, default=4, help='if no set partitions are defined then we use partitions [1,..,n_patches], otherwise it defines the number of tokens that auxiliary data representations are squeezed into')
    parser.add_argument('--resolution_set', nargs='+', default=[], help='iterable defining the partitions of the input sequence, dominates n_patches')
    parser.add_argument('--revin', type=int, default=1, choices = [0,1], help='Boolean (1 or 0 encoded), include instance normalisation (subtract last and affine)')
    parser.add_argument('--norm', type=str, default="BatchNorm", choices = ["BatchNorm", "LayerNorm", "none"], help='Normalization option')
    parser.add_argument('--activation', type=str, default="ReLU", choices = ["ReLU", "GeLU"], help='Activation function')
    parser.add_argument('--batch_size', type=int, default= 640 , help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type = int, default = 20, help = "number of epochs trained for")
    parser.add_argument('--patience', type = int, default = 3, help = "number of epochs of no validation improvement before training terminated")
    parser.add_argument('--folder_path',type = str, help = "folder path")
    parser.add_argument('--start_year_week',type = str, help = "Needs to be in the format 'year-week', the domain of this depends on the dataset, ex: '2021-50'")
    parser.add_argument('--end_year_week',type = str, help = "Needs to be in the format 'year-week', the domain of this depends on the dataset, ex: '2021-50'")
    parser.add_argument('--p_val', default = 0.15,type = float, help = "rough proportion (dates are sorted, then rounding is done on number of days that should be in val, and it will never round down to 0) of dataset for validation")
    parser.add_argument('--p_test', default = 0.2,type = float, help = "rough proportion (dates are sorted, then rounding is done on number of days that should be in test, and it will never round down to 0) of dataset for test")
    parser.add_argument('--keep_longest', default = 0.1, type = float, help = 'keep the top quantile of values by length defined by this variable')


    args = parser.parse_args()

    return args


def main():
    
    # Take arguments from shell file
    args = parse_args()

    # Configure optimiser and losses that are measured
    optimizer = torch.optim.Adam
    logging_metrics = [torch.nn.MSELoss(),torch.nn.L1Loss()]

    if args.loss == "MAE":
        loss_fn = torch.nn.L1Loss()
    elif args.loss == "RMSE":
        loss_fn = torch.nn.MSELoss()
    elif args.loss == "Smooth":
        loss_fn = torch.nn.SmoothL1Loss()

    # Set experiment configuration, somewhat outdated as most arguments now fed manually
    config = ExperimentConfig(args.experiment)

    # Load and process pricing data (code not included due to commercial sensitivity)

    # full_df = load_df_pricing(folder_path = args.folder_path, start = args.start_year_week, end = args.end_year_week)
    # full_df = preprocess_pricing_dataset(full_df)

    # Quantize dataset, define cutoff value with keep_quantile_longest_sequences (eg 0.95 indicates only 0.05 longest)
    # dataset = quantize_by_row(full_df,keep_quantile_longest_sequences = 1 - args.keep_longest)

    # Set dataformatter which takes care of scaling and provides instructions for dataset piping
    data_formatter = config.make_data_formatter()

    # Specific to pricing data as series are uniquely defined by store id x product id x due date
    data_formatter.sequence_id = 'sequence_id'

    # Fixed forecast horizon, lookback horizon set to longest in dataset (in practice the lookback horizon should be slightly inflated beyond this)
    forecast_horizon = args.forecast_horizon
    data_formatter._params['forecast_horizon'] = forecast_horizon
    # Define lookback horizon relative to the length of the longest sequence (the preprocessing code calculates the length of every sequence ad saves it in sequence_len)
    max_total_len = dataset.sequence_len.max()
    lookback_horizon = max_total_len - data_formatter._params['forecast_horizon']
    data_formatter._params['lookback_horizon'] = lookback_horizon

    ## Add variables to model
    # Add global variables
    data_formatter._column_definition += [
        # Static global variables
        ('product group', DataTypes.CATEGORICAL, InputTypes.STATIC_GLOBAL),
        ('brand', DataTypes.CATEGORICAL, InputTypes.STATIC_GLOBAL),
        ('stock', DataTypes.REAL_VALUED, InputTypes.STATIC_GLOBAL),
    
    
        # Known global temporal 
        ('iteration of price reduction', DataTypes.CATEGORICAL, InputTypes.KNOWN_TEMPORAL_GLOBAL),
        ('proportion of stock marked dow', DataTypes.REAL_VALUED,InputTypes.KNOWN_TEMPORAL_GLOBAL),
    
    ]
    
    # Add time features (can be turned off)
    # Time known global temporal (only works if time features are added)
    if args.time_features:
        data_formatter._column_definition += [
            ('hour', DataTypes.REAL_VALUED,InputTypes.KNOWN_TEMPORAL_GLOBAL),
            ('day_of_week', DataTypes.CATEGORICAL,InputTypes.KNOWN_TEMPORAL_GLOBAL),
            ('month',DataTypes.CATEGORICAL,InputTypes.KNOWN_TEMPORAL_GLOBAL),
        ]
    # Add specific variables
    # Channel specific variables have to be of the form (f'{channel}__{variable_name}', DataTypes.REAL_VALUED, InputTypes.KNOWN_TEMPORAL_SPECIFIC or InputTypes.STATIC_SPECIFIC)
    for tar in data_formatter.get_target:
        # Known termporal specific
        data_formatter._column_definition += [
        (f'{tar}__price', DataTypes.REAL_VALUED, InputTypes.KNOWN_TEMPORAL_SPECIFIC),
    ]
        # Static specific variables
        data_formatter._column_definition += [
        (f'{tar}__channel_id', DataTypes.CATEGORICAL, InputTypes.STATIC_SPECIFIC),
    ]

    # Extract number of channels (as defined by TARGET input types)
    targets = data_formatter.get_target
    n_out = len(targets)
    # Add columns to preserve unscaled target values (lazy)
    for tar in targets:
        dataset[f'{tar}_original'] = dataset[tar]

    # Determine date cutoffs based on proportion of dataset that should be used as validation and test sets
    splits = determine_cutoffs(df = dataset, date_col = data_formatter.get_date_col, p_val = args.p_val, p_test = args.p_test)
    # Split data, also does scaling
    train, valid, test = data_formatter.split_data_pricing(dataset,splits = splits)
    
    # String tracking the configuration of the experiment
    experiment_string = f"""
    The following variables are defined:
    * path = {args.folder_path}
    * start_year_week = {args.start_year_week}
    * start_year_week = {args.end_year_week}
    * keep_longest = {args.keep_longest}
    * experiment = {args.experiment}
    * forecast_horizon = {args.forecast_horizon}
    * lookback_horizon = {lookback_horizon}
    * time_features = {args.time_features}
    * loss = {loss_fn}
    * d_model = {args.d_model}
    * d_ff = {args.d_ff}
    * d_mixer = {args.d_mixer}
    * n_head = {args.n_head}
    * N = {args.N}
    * dropout = {args.dropout}
    * ablation_options = {args.ablation_options}
    * densest_partition = {args.densest_partition}
    * resolution_set = {args.resolution_set}
    * n_out = {n_out}
    * revin = {args.revin}
    * norm = {args.norm}
    * batch_size = {args.batch_size}
    * optimizer = {optimizer}
    * learning_rate = {args.learning_rate}
    """
    print(experiment_string)

    # Set and configure torch datasets and dataloaders
    training = ForecastingDatasetPricing(train,data_formatter)
    train_dataloader = torch.utils.data.DataLoader(training,batch_size=args.batch_size, shuffle = True)

    validation = ForecastingDatasetPricing(valid,data_formatter)
    val_dataloader = torch.utils.data.DataLoader(validation,batch_size=args.batch_size, shuffle = False)

    testing = ForecastingDatasetPricing(test,data_formatter)
    test_dataloader = torch.utils.data.DataLoader(testing,batch_size=args.batch_size, shuffle = False)

    # Construct model based on the shape of the inputs as defined by the data_formatter and the hyperparameters from args
    model = LightningMultipleResolutionTokenization(
                formatter = data_formatter,

                d_model = args.d_model,
                d_ff = args.d_ff,
                d_mixer = args.d_mixer,
                n_head = args.n_head,
                N = args.N,
                n_out = n_out,

                ablation_options = args.ablation_options,
                densest_partition = args.densest_partition,
                set_partitions = args.resolution_set,

                dropout=args.dropout,
                revin = args.revin,
                norm = args.norm,
                activation = args.activation,


                logging_metrics = logging_metrics,
                loss = loss_fn,
                optimizer_torch = optimizer,
                learning_rate = args.learning_rate
                 )

    # Lighnting callbacks
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss",)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    # Training module and process
    exp_name = f'{args.experiment}_from_{args.start_year_week}_to_{args.end_year_week}'
    trainer = pl.Trainer(accelerator="cpu",
                         max_epochs = args.epochs,
                         callbacks = [checkpoint_callback,early_stop_callback],
                         logger = CSVLogger(save_dir = f"logger/{exp_name}_f{forecast_horizon}l{lookback_horizon}/"),
                        )

    trainer.fit(model, train_dataloader, val_dataloader)

    ## Reporting

    # add a dictionary for conditional splits
    split_dict = dict(zip(['val_cutoff','test_cutoff'],splits))

    test_scores = trainer.test(model, test_dataloader,ckpt_path='best')
    # report experiment parameters
    report = {**vars(args),**split_dict}
    report['lookback_horizon'] = lookback_horizon

    # add test scores
    report = {**report,**test_scores[0]}
    
    ######## Calculate losses for full price sales for MRT and internal predictions
    other_evals = {}
    original_columns = [f'{tar} original' for tar in targets]
    predict_columns = [f'{tar} predict' for tar in targets]
    
    predict = trainer.predict(model,test_dataloader)
    predictions = torch.concat([p[0] for p in predict],dim=0)
    sequence_ids = sum([p[1] for p in predict],[])
    predict_dct = {sequence:predictions[i,...] for i, sequence in enumerate(sequence_ids)}

    # Add predictions from model by sequence id
    test_horizons = []
    for name, group in test.groupby('sequence_id'):
        sub_df = group.sort_values('per_series_index')[-forecast_horizon:]
        sub_df[predict_columns] = predict_dct[name].reshape(-1,len(targets))
        test_horizons.append(sub_df)
    
    test_horizons = pd.concat(test_horizons)

    # Extract tensors from dataframe
    true = torch.tensor(test_horizons['full price sales original'].values).view(-1,forecast_horizon,1)
    pred_internal = torch.tensor(test_horizons['internally predicted full price sales'].values).view(-1,forecast_horizon,1)
    pred_deep = torch.tensor(test_horizons['full price sales predict'].values).view(-1,forecast_horizon,1)

    # Calculate losses for all metrics
    for metric in model.logging_metrics:
        name = str(type(metric)).split("'>")[0].split('.')[-1]
    
        # internal eval
        internal = metric(true,pred_internal)
        other_evals[f"internal_test_loss_full_price_{name}"] = float(internal)
        # deep eval
        deep = metric(true,pred_deep)
        other_evals[f"MRT_test_loss_full_price_{name}"] = float(deep)

    # add comparative evals for full price forecasts
    report = {**report,**other_evals}
    ########
    
    report_path = f"./{report_file}"

    file_exists = os.path.exists(report_path)

    writer = CsvDictWriter()

    if file_exists:
        existing_report = pd.read_csv(report_path)
        for dct in existing_report.to_dict("records"):
            writer.add_dict(dct)

    writer.add_dict(report)
    writer.write(report_path)

if __name__ == '__main__':
    main()
