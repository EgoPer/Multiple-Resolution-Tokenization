import numpy as np
import pandas as pd
from utils import get_single_col_by_input_type, InputTypes, DataTypes
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import copy
from torch import nn
import torch


def get_cols_corresponding_to_input_and_data_type(column_def, input_type, data_type):
    """
    Additional utility that extracts the columns corresponding to a specific input and data type combination
    """
    return [col[0] for col in column_def if col[2] == input_type and col[1] == data_type]



def get_inverse_transform_many_scalers(inp, pred, formatter):
    """
    Reverts the target and prediction to original scale.
    """
    tar = inp[1][0].unsqueeze(-1)

    ids = [list(formatter._target_scaler)[inp[0]["groups"][i]] for i, p  in enumerate(pred)]
    predn = torch.tensor(np.stack([formatter._target_scaler[ids[i]].inverse_transform(value.detach())
          for i, value in enumerate(pred)]))
    tarn = torch.tensor(np.stack([formatter._target_scaler[ids[i]].inverse_transform(value.detach())
          for i, value in enumerate(tar)]))
    return predn, tarn



def get_names_multi(formatter):
    """
    In use.
    Returns a dictionary with the names corresponding to different input and data type combinations.
    """
    column_definition = formatter._column_definition

    column_subsets = {
        "target" : formatter.get_target,
        "s_continuous_specific" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.STATIC_SPECIFIC, DataTypes.REAL_VALUED),
        "s_categorical_specific" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.STATIC_SPECIFIC, DataTypes.CATEGORICAL),
        "s_continuous_global" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.STATIC_GLOBAL, DataTypes.REAL_VALUED),
        "s_categorical_global" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.STATIC_GLOBAL, DataTypes.CATEGORICAL),
        "x_continuous_specific" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.KNOWN_TEMPORAL_SPECIFIC, DataTypes.REAL_VALUED),
        "x_categorical_specific" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.KNOWN_TEMPORAL_SPECIFIC, DataTypes.CATEGORICAL),
        "x_continuous_global" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.KNOWN_TEMPORAL_GLOBAL, DataTypes.REAL_VALUED),
        "x_categorical_global" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.KNOWN_TEMPORAL_GLOBAL, DataTypes.CATEGORICAL),
        "y_others_continuous" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.OBSERVED_INPUT, DataTypes.REAL_VALUED),
        "ensemble_continuous" : get_cols_corresponding_to_input_and_data_type(column_definition, InputTypes.ENSEMBLE, DataTypes.REAL_VALUED),
    }
    def sort_subset_to_match_target_channels(subset, target_order):
        variables = np.unique([column.split("__")[-1] for column in subset if "__" in column])
        # this makes sure they are sorted by variable first, and channel order second
        # Each n_channel indices will therefore correspond to the order of channels
        subset_sorted = [f"{channel}__{variable}" for variable in variables for channel in target_order]
        return subset_sorted

    # Indices for specific auxiliary information have to be arranged so the channels are consistent
    # EG the prices for a channel have to match the target channel
    # This step mitigates mistakes in the column definion sequence
    # Channel specific variables have to be of the form f"{channel__variablename}

    target_order = column_subsets["target"]

    for key, subset in column_subsets.items():
        if "specific" in key:
            subset_sorted = sort_subset_to_match_target_channels(subset, target_order)
            column_subsets[key] = subset_sorted

    return column_subsets


def get_counts_multi(formatter):
    """
    Return the counts of variables in a setting which includes channel specific variables.

    Needed to define facets of the TVKT module.
    """
    column_definition = formatter._column_definition
    cat_cols = [c[0] for c in column_definition if (c[2] != InputTypes.ID and c[2] != InputTypes.TIME and c[1] == DataTypes.CATEGORICAL)]
    category_count_dct = formatter._num_classes_per_cat_input

    names = get_names_multi(formatter)
    sizes_dct = {}

    for key in names.keys():
        if "categorical" in key:
            keys = names[key]
            if "specific" in key:
                variables = np.unique([column.split("__")[-1] for column in names[key] if "__" in column])
                maximum_category_count_by_variable = [np.max([category_count_dct[column]
                                                              for column in names[key] if var in column])
                                                      for var in variables]
                sizes_dct[key] = maximum_category_count_by_variable

            else:
                sizes_dct[key] = [category_count_dct[key] for key in keys if key in category_count_dct.keys()]
        elif ("continuous" in key) or ("target" in key):
            sizes = len(names[key])
            if "specific" in key:
                variables = np.unique([column.split("__")[-1] for column in names[key] if "__" in column])
                sizes_dct[key] = int(len(variables))
            else:
                sizes_dct[key] = sizes

    return sizes_dct


def get_indices_data_loader(
                formatter,
                columns
                ):

    
    """
    Utility for extracting column indices in the dataloader
    """
    column_definition = formatter._column_definition

    column_subsets = get_names_multi(formatter)

    #get indices
    def return_indices(columns,subset):
        column_index_dct = dict(zip(columns,range(len(columns))))

        return [column_index_dct[column] for column in subset]

    indices = {}


    for key, subset in column_subsets.items():

        indices[key] = return_indices(columns,subset)

    return indices


