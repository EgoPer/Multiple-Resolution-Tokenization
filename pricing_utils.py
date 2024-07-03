import numpy as np
import torch
import os
import pandas as pd
from pathlib import Path, PurePosixPath
from datetime import datetime
import torch
from training_utils import get_indices_data_loader
from torch.nn.functional import pad
    

def create_starts(start, finish, shutbetween = [24,5]):
    """
    Utility returning start times for quantisation. Assumes everything is open 0500-0000, not sure it would pass tests with openings past 0000.
    """
    delta = (finish-start).total_seconds() / 3600
    time_deltas = [pd.Timedelta(hours = int(stamp/2),minutes = int(60*(stamp/2-int(stamp/2)))) for stamp in range(int(delta*2 + 1))]
    starts = ([start + time_deltas[i] for i in range(len(time_deltas)-1)])
    starts = [td for td in starts if ((td.hour+ td.minute/60)<shutbetween[0] and (td.hour+ td.minute/60)>=shutbetween[1])]
    return starts
    
def add_time_features_pricing(dataset,date_col):
    s = dataset.copy()
    date = pd.to_datetime(s[date_col])
    s["hour"] = date.dt.hour + date.dt.minute/60
    s["day_of_week"] = date.dt.dayofweek
    s["day_of_month"] = date.dt.day
    s["month"] = date.dt.month
    s["week_of_year"] = date.dt.isocalendar().week
    return s

def round_except_0(float):
    """
    rounding used in holdout partitions
    """
    rounded = round(float)
    if rounded > 0:
        return rounded
    else:
        return 1

def determine_cutoffs(df, date_col,p_val,p_test):
    """
    determine cutoff dates for holdout partitions
    """
    sorted_dates = np.sort(pd.to_datetime(df[date_col]).dt.date.unique())

    n_dates = len(sorted_dates)
    n_test = round_except_0(n_dates * p_test)
    n_val = round_except_0(n_dates * p_val)


    val_cutoff = sorted_dates[-(n_val + n_test)]
    test_cutoff = sorted_dates[-n_test]

    return (val_cutoff,test_cutoff)

        # ('group id for normalisation', DataTypes.REAL_VALUED, InputTypes.ID),
        # ('start of sequence date', DataTypes.REAL_VALUED, InputTypes.TIME),
        # ('sold full price', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # ('sold reduced price', DataTypes.REAL_VALUED, InputTypes.TARGET),
def quantize_by_row(dataframe, 
                    columns_to_quantize = ['full price sales','reduced price sales','internally predicted full price sales'],
                    id_columns = ['due date', 'store id', 'product id'],
                    keep_quantile_longest_sequences = 0.9,
                    time_features = True
                   ):
    """
    Quantisation code.
    """
    dataframe = dataframe.copy()
    # Round the starts and ends of sequences to 30 minute resolution
    dataframe.start_dt = dataframe.start_dt.round('30min')
    dataframe.end_dt = dataframe.end_dt.round('30min')
    dataframe['start_of_sequence_date'] = dataframe.groupby(id_columns).start_dt.transform(lambda x: x.dt.date.min())
    # Calculate lengths of sequences in hours
    dataframe['sequence_length_hours'] = (dataframe.groupby(id_columns).end_dt.transform('max') - dataframe.groupby(id_columns).start_dt.transform('min')).dt.total_seconds()/3600

    
    # create starts of quantized time windows
    dataframe['qstarts'] = dataframe.apply(lambda x: create_starts(x.start_dt,x.end_dt),axis = 1)
    # calculate lengths of subsequences for value quantisation
    dataframe['subsequence_len'] = dataframe.qstarts.apply(len)
    dataframe = dataframe[dataframe['subsequence_len']>0]
    # Adjust values so they reflect average activity during this stage
    for column in columns_to_quantize:
        dataframe[column] = dataframe.apply(lambda x: x[column]/x.subsequence_len,axis = 1)
        
    # Calculate lengths of sequences in steps
    dataframe['sequence_len'] = dataframe.groupby(id_columns).subsequence_len.transform(lambda x: x.sum())
    # Set lower bound on sequence length
    quantile = dataframe['sequence_len'].quantile(keep_quantile_longest_sequences)
    # Filter by lower bound
    dataframe = dataframe[dataframe['sequence_len']>=quantile]
    
    # Explode dataframe so that start values are by row
    dataframe = dataframe.explode('qstarts')
    # Add ends of windows for completeness
    dataframe['qends'] = dataframe['qstarts'] + pd.Timedelta(minutes = 30)
    
    # Create index of subsequence for each product x store x due-date
    ndf = []
    for name, group in dataframe.groupby(id_columns):
        group = group.sort_values(by = 'qstarts').reset_index(drop=True)
        group['per_series_index'] = group.index
        ndf.append(group)
    
    ndf  = pd.concat(ndf).reset_index(drop=True)
    if time_features:
        ndf = add_time_features_pricing(ndf, 'qstarts')
    return ndf


class ForecastingDatasetPricing(torch.utils.data.Dataset):
    """
    Torch dataset which works with pricing data. Requires a lot of delicate piping.

    Works on dataframes which have sequence_id, each unique sequence is sorted by time and each row contains the context and observation at a single timestep.

    can accommodate varying length sequences by padding them
    """
    def __init__(self,
                 df,
                 formatter,
                ):
        self.df = df

        self.df_columns = df.columns


        self.formatter = formatter
        self.params = self.formatter._params

        self.group_id = self.formatter.get_group_id
        self.date_col = self.formatter.get_date_col
        self.targets = self.formatter.get_target
        self.n_channels = len(self.targets)

        self.sequence_id = self.formatter.sequence_id
        self.index_to_sequence_id_dct = dict(zip(range(self.df[self.sequence_id].unique().shape[0]),self.df[self.sequence_id].unique()))
        
        self.lookback_horizon = self.params["lookback_horizon"]
        self.forecast_horizon = self.params["forecast_horizon"]
        self.max_sequence_length = self.lookback_horizon + self.forecast_horizon

        self.column_definition = self.formatter._column_definition

        self.indices = get_indices_data_loader(self.formatter,self.df_columns)

        

    def adapt_dtype(self,x,name):
        if "categorical" in name:
            return x.long()
        else:
            return x.float()

    def pad_series(self,x, pad_size):
        if x.shape[-1] > 0:
            # left pad 0 dimension by pad size
            pad_shape = [0]*2*(len(x.shape)-1) + [pad_size,0]
            pad_shape = tuple(pad_shape)
            x = pad(x,pad_shape)
        else:
            shape = list(x.shape)
            shape[0] += pad_size
            x = x.view(shape)
        return x
        
    def __len__(self):
        return self.df[self.sequence_id].unique().shape[0]

    def __getitem__(self, index):

        data = {}
        sequence_id = self.index_to_sequence_id_dct[index]
        sub_df = self.df[(self.df['sequence_id']==sequence_id)]
        pad_size = self.max_sequence_length - sub_df.shape[0]

        data['group_id'] = sub_df[self.group_id].iloc[0]
        data['sequence_id'] = sequence_id
        for name in self.indices.keys():

            series = self.adapt_dtype(torch.tensor(sub_df.iloc[:,self.indices[name]].values),name)
            series = self.pad_series(series,pad_size)

            if "specific" in name:
                # [(b,)time,variable,channel]
                series = series.view([self.max_sequence_length,-1,self.n_channels])


            past = series[:-self.forecast_horizon]

            # deal with static variables
            if name.startswith("s_c"):
                if 'specific' in name:
                    # [(b,)channel,time,variable]
                    past = series.permute([-1,-3,-2])
                    past = past[:,-1,...]
                    data[f"{name}"] = past
                else:
                    past = series[-1,...]
                    data[f"{name}"] = past
            else:
                if "specific" in name:
                    # [(b,)channel,time,variable]
                    past = past.permute([-1,-3,-2])

                data[f"{name}_past"] = past

            # deal with future values
            if (name.startswith("x_c")) or ("target" in name):
                future = series[-self.forecast_horizon:]

                if "specific" in name:
                    # [(b,)channel,time,variable]
                    future = future.permute([-1,-3,-2])

                data[f"{name}_future"] = future


        return data