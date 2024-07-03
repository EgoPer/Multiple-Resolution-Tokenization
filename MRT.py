import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import MultiHeadAttention
import lightning.pytorch as pl
from typing import List
from layers.RevIN import RevIN
from training_utils import get_counts_multi
from layers.utils import Transpose, get_activation_fn


class PositionWiseFeedForward(nn.Module):
    """
    A simple double feedforward with dropout. Used in transformer blocks after attention. 
    It is applied to [B,...,T,L] - mixing the latent dimension of every token
    """
    def __init__(self, d_model, d_ff, dropout,activation):
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed forward layer.
        x: [batch, channel, token, latent]
        out: [batch, channel, token, latent]
        """
        x = self.W_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.W_2(x)
        return x

class CategoricalEmbedding(nn.Module):
    """
    Entity embedding is a layer which learns embeddings for categorical variables.
    It is applied to [B,C,T] so directly to variables.
    Essentially just learns a matrix where each column corresponds to an embedding for each instance for each categorical variable.
    Special values: 0 category label indicates deactivation - useful for varying length inputs as no representations or learning is done when inputs do not exist (as opposed to not being available which corresponds to category label 1)
    """
    def __init__(self, categorical_feature_size_list, d_model):
        super(CategoricalEmbedding, self).__init__()

        # Note that the 0 category will always be seen as padding, not producing an embedding or gradients, adjust preprocessing accordingly 
        self.embedding_layer = nn.ModuleList([nn.Embedding(categorical_feature_size_list[i], d_model,padding_idx=0) for i in range(len(categorical_feature_size_list))])
        self.size = len(categorical_feature_size_list)

        self.dummy_parameter = nn.Parameter(torch.empty(0))
    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, x = torch.tensor([])):
        """
        Forward pass through the entity embedding.
        x: [batch, channel, time, variable]
        output: [batch, channel, time, variable, latent]
        """
        if self.size != 0:
            x = x.unsqueeze(-2)

            outputs = []
            for i, embedding in enumerate(self.embedding_layer):
                ee = embedding(x[...,i])
                outputs.append(ee)
            return torch.cat(outputs,dim = -2)
        else:
            empty = torch.tensor([]).to(self.device)
            return empty

class NumericalEmbedding(nn.Module):
    """
    Numerical embedding is a layer which learns embeddings for numerical variables. 
    It is applied directly to [B,C,T], so directly to variables.
    Learns a multilinear embedding for each scalar variable. A value of 0 indicates that the network should be disactivated at that point (for varying length inputs).
    A separate embedding is learned for nans.
    """
    def __init__(self, num_continuous_variables, d_model,n_in = 1):
        super(NumericalEmbedding, self).__init__()
        self.embedding_layer = nn.ModuleList([nn.Linear(n_in, d_model,bias = False) for _ in range(num_continuous_variables)])
        # If the entry is nan a special embedding is employed
        self.nan_embeddings = [nn.Parameter(torch.rand(1,d_model)) for _ in range(num_continuous_variables)]
        self.size = num_continuous_variables

        self.dummy_parameter = nn.Parameter(torch.empty(0))
    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, x = torch.tensor([])):
        """
        Forward pass through the numerical embedding.
        x: [batch, channel, time, variable]
        output: [batch, channel, time, variable, latent]
        """

        if self.size != 0:
            x = x.unsqueeze(-2)
            outputs = []
            for i, embedding in enumerate(self.embedding_layer):
                ne = embedding(x[...,i]).unsqueeze(-2)
                nan_bool = torch.isnan(ne).all(axis = -1)
                ne[nan_bool] = self.nan_embeddings[i]
                outputs.append(ne)
            return torch.cat(outputs,dim = -2)
        else:
            empty = torch.tensor([]).to(self.device)
            return empty

class EncoderLayer(nn.Module):
    """
    Encoder layer consists of self-attention, and feed forward layer with
    two skip connections and layer normalisation (add & norm).
    """
    def __init__(self, d_model, d_ff, n_head, dropout,norm, activation):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_model, n_head)
        self.mlp = PositionWiseFeedForward(d_model = d_model,d_ff = d_ff,dropout = dropout,activation = activation)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fcn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.ModuleList([nn.Sequential(Transpose(-1,-2), nn.BatchNorm1d(d_model), Transpose(-1,-2))
                            for _ in range(2)])
        elif "layer" in norm.lower():
            self.norm_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        else:
            self.norm_attn = nn.ModuleList([nn.Identity() for _ in range(2)])

    def forward(self, E, mask=None):
        """
        Forward pass through the encoder layer with skip connection and layer normalization.
        x: [batch, token, latent]
        mask: [batch, token, latent]
        """
        # skip connections only work if the input has already been projected to d_model
        Ea = self.self_attn(E, E, E, mask)
        Ea = self.dropout_attn(Ea)
        Ehat = self.norm_attn[0](Ea + E)     # Norm

        Etilde = self.mlp(Ehat)       # Mlp and skip connection
        Etilde = self.dropout_fcn(Etilde)
        Etilde = self.norm_attn[1](Etilde + Ehat)     # Norm

        return Etilde


class Encoder(nn.Module):
    """
    Encoder is a stack of N encoder layers.
    """
    def __init__(self,d_model,d_ff,n_head, N, dropout,norm, activation):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model = d_model, d_ff = d_ff, n_head = n_head, dropout = dropout,norm = norm, activation = activation) for _ in range(N)])

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder.
        x: [batch, token, latent]
        mask: [batch, token, latent]
        """
        outputs = []
        for layer in self.encoder_layers:
            x = layer(x, mask)
            outputs.append(x)
        return outputs


def split_into_k_parts(iterable, k):
    """
    Utility for multiple resolution patching, splits an iterable into k parts, the first remainder many parts are 1 longer
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    L = iterable.shape[-1]
    if L < k:
        raise ValueError("The iterable has fewer elements than k")

    avg = L // k
    remainder = L % k

    start = 0
    for i in range(k):
        end = start + avg + (1 if i < remainder else 0)
        yield iterable[...,start:end]
        start = end

def len_of_split_into_k_parts(L, k):
    """
    L - length of iterable
    k - number of parts to split into
    Utility for multiple resolution patching, provides the length of each of the split parts.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if L < k:
        raise ValueError("The iterable has fewer elements than k")

    avg = L // k
    remainder = L % k

    return [avg + (1 if i < remainder else 0) for i in range(k)]

class Splitter(nn.Module):
    """
    Base splitter module for multiple resolutions, only handles one resolution.
    Operates over past time series data, embeddings extracted from patches.
    Shared weights, left padding for shorter patches.

    shared indicates that we use a single set of weights for all patches.
    """
    def __init__(self, d_model, lookback_horizon, number_of_splits, shared):
        super(Splitter, self).__init__()
        self.shared = shared
        self.number_of_splits = number_of_splits

        self.split_lengths = len_of_split_into_k_parts(lookback_horizon,self.number_of_splits)
        self.partitioner = split_into_k_parts

        if self.shared:
            max_len = int(np.max(self.split_lengths))
            self.pads = [max_len - l for l in self.split_lengths]
            self.split_projections = nn.Linear(max_len, d_model)
        else:
            self.split_projections = nn.ModuleList([nn.Linear(lenght, d_model) for lenght in self.split_lengths])

    def forward(self, x):
        """
        input: [batch,channel,time]
        ouptut: [batch,channel,token,latent]
        """
        projections = []
        for i, part in enumerate(self.partitioner(x,self.number_of_splits)):
            if self.shared:
                pad = torch.zeros(part.size(0),part.size(1),self.pads[i])
                part = torch.cat([pad,part],dim=-1)
                projection = self.split_projections(part)
            else:
                projection = self.split_projections[i](part)
            projections.append(projection)

        projections = torch.stack(projections,dim = 2)

        return projections


class ManySplitters(nn.Module):
    """
    Module which combines many splitter modules to transform a time series into a collection of tokens extracted at different resolutions (as defined by the iterable partitions).
    shared indicates that each resolution only learns one set of weights for all patches
    """
    def __init__(self, d_model, lookback_horizon, partitions, shared = True):
        super(ManySplitters, self).__init__()

        self.splitters = nn.ModuleList([Splitter(d_model = d_model,lookback_horizon = lookback_horizon,number_of_splits = number_of_splits, shared = shared)
                                      for number_of_splits in partitions])

    def forward(self,x):
        """
        input: [batch,channel,time]
        out: [batch, channel, token, latent]
        """
        projections = []
        for i, splitter in enumerate(self.splitters):
            projection = splitter(x)
            projections.append(projection)

        projections = torch.cat(projections,dim = 2)

        return projections


class SplitterBasisComination(nn.Module):
    """
    Splits time series embeddings into patches and learns a basis combination.
    This is for a single resolution.
    Operates on [B,C,T,L], splits T into number_of_splits matrices and combines columns.
    shared indicates that each resolution only learns one set of weights for all patches
    """
    def __init__(self, d_model, horizon, number_of_splits, shared=True):
        super(SplitterBasisComination, self).__init__()
        self.shared = shared
        self.number_of_splits = number_of_splits

        self.split_lengths = len_of_split_into_k_parts(horizon,self.number_of_splits)
        self.partitioner = split_into_k_parts

        max_len = int(np.max(self.split_lengths))
        self.pads = [max_len - l for l in self.split_lengths]
        self.split_projections = nn.Parameter(torch.rand(max_len).unsqueeze(-1))

    def forward(self, x):
        """
        input: [batch,channel, time, latent]
        ouptut: [batch,channel, token, latent]
        """
        projections = []
        x = x.transpose(-1,-2) # [batch,channel,latent, horizon]
        for i, part in enumerate(self.partitioner(x,self.number_of_splits)):


            # part # [batch,channel,latent, horizon split]
            combination = part @ self.split_projections[:part.shape[-1],...] # [batch,channel,latent, 1]

            combination = combination.transpose(-1,-2) # [batch,channel,1,latent]

            projections.append(combination)

        projections = torch.concat(projections,dim = -2) # [batch,channel,n_splits,latent]

        return projections

class ManySplittersBasisComination(nn.Module):
    """
    Combines many splitter basis combination modules into single multiple resolution processing module
    
    shared indicates that each resolution only learns one set of weights for all patches
    """
    def __init__(self, d_model, horizon, partitions, shared =True):
        super(ManySplittersBasisComination, self).__init__()


        self.splitters = nn.ModuleList([SplitterBasisComination(d_model = d_model,horizon = horizon,number_of_splits = number_of_splits)
                                      for number_of_splits in partitions])

    def forward(self,x):
        """
        input: [batch,channel, time, latent]
        ouptut: [batch,channel, token, latent]
        """
        projections = []
        for i, splitter in enumerate(self.splitters):
            projection = splitter(x)
            projections.append(projection)

        projections = torch.cat(projections,dim = -2)

        return projections


class AdjustedEncoder(nn.Module):
    """
    Encoder wrapper which enables multiple channels/channel independence.
    Needed as the code for multi-head attention is not fully abstracted. This is the result of incremental work and improvements.
    Operates over [B,C,T,L] by iterating and encoder module over C (same encoder weights for all)
    """
    def __init__(self, d_model, d_ff, n_head, N, dropout, norm, ablation_options=[], activation = 'gelu'):
        super(AdjustedEncoder, self).__init__()
        self.N = N
        self.encoder = Encoder(d_model = d_model,d_ff = d_ff,n_head = n_head, N = N, dropout = dropout,norm = norm, activation = activation)
    def forward(self, x):
        """
        input: [batch,channel, token, latent]
        ouptut: [batch,channel, token, latent]
        """
        number_of_channels = x.size()[1]

        out = []

        for i in range(number_of_channels):
            encoder_out = self.encoder(x[:,i,...]) # list([batch,1,latent channel, hidden])
            out.append(encoder_out)

        out = [torch.stack([out[i][n] for i in range(number_of_channels)],dim = 1) for n in range(self.N)]

        return out

class AdjustedMixer(nn.Module):
    """
    This module is designed to extract cross series representations from the encoder input matrix.
    This is achieved by mixing across the token and channel dimension.
    In addition the number of tokens is squeezed to densest_partition and all the channels are squeezed into one after mixing.
    The one channel feature is done so that all channels then share the same cross-series embeddings.

    This module takes what would have been the input for the encoder, 
    this means that when auxiliary tokens are included the n_splitter_embeddings has to be adjusted accordingly.
    """
    def __init__(self,n_channels,n_splitter_embeddings,d_model,norm,dropout,activation, densest_partition, squeeze_joint_patch_channels = True):
        super(AdjustedMixer,self).__init__()

        self.n_splitter_embeddings = n_splitter_embeddings
        self.d_model = d_model
        self.n_channels = n_channels
        self.squeeze_joint_patch_channels = squeeze_joint_patch_channels

        if "batch" in norm.lower():
            self.norm_attn = [nn.BatchNorm1d(self.n_splitter_embeddings) for _ in range(2)]
        if "layer" in norm.lower():
            self.norm_attn = [nn.LayerNorm(self.n_splitter_embeddings) for _ in range(2)]
        else:
            self.norm_attn = [nn.Identity() for _ in range(2)]

        self.time_mix = nn.Linear(self.n_splitter_embeddings,self.n_splitter_embeddings)

        if self.squeeze_joint_patch_channels:
            self.squeeze_joint_patches_layer = nn.Linear(self.n_splitter_embeddings,densest_partition)

        self.feature_mix_1 = nn.Linear(self.n_channels,self.d_model)
        self.feature_mix_2 = nn.Linear(self.d_model,self.n_channels)

        self.squeeze_channels_layer = nn.Linear(self.n_channels,1) # Have same token representation across channels

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)


    def forward(self,input):
        """
        input: [batch,channel, token, hidden]
        output: [batch,channel, token, hidden]
        """
        # time mixing
        x = self.norm_attn[0](input)
        x = Transpose(-1,-2)(x) #[batch, channel, hidden, token]
        x = self.time_mix(x) # mix across divider tokens
        x = self.activation(x)
        x = self.dropout(x)
        x = Transpose(-1,-2)(x) #[batch,channel, token, hidden]

        res = x + input

        if self.squeeze_joint_patch_channels:
            res = Transpose(-1,-2)(res)
            res = self.squeeze_joint_patches_layer(res) # #[batch,channel, token = deepest_partition, hidden]
            res = Transpose(-1,-2)(res)

        # cross series information extraction
        x = self.norm_attn[1](res)
        x = Transpose(-1,-3)(x) #[batch,hidden, token, channel]
        x = self.feature_mix_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.feature_mix_2(x)
        x = self.dropout(x)
        x = Transpose(-1,-3)(x) #[batch,channel, token, hidden]
        x = x + res

        #squeeze down to one channel so cross series representations the same for all channels
        x = Transpose(-1,-3)(x)
        x = self.squeeze_channels_layer(x) #[batch, 1, token, hidden]
        x = Transpose(-1,-3)(x)

        return x

class OutputLayerReverseSegments(nn.Module):
    """
    New output layer which learns to project the tokenised representations of segments into future segments.
    This module performs a sort of reverse splitting.
    It operates on [B, C, T, L] -> [B, C, T = f].
    This is achieved by taking sequences of tokens from their corresponding splitter positions and projecting them into proportional blocks of the forecast at each resolution as defined in the partitions. The forecasts are then summed across resolutions.
    """
    def __init__(self, d_model, partitions, forecast_horizon, shared = True):
        super(OutputLayerReverseSegments, self).__init__()

        self.partitions = partitions

        self.parition_indices = [0] + [int(np.sum(self.partitions[:i+1])) for i, partition in enumerate(self.partitions)]

        self.splits = [len_of_split_into_k_parts(L = forecast_horizon, k = partition)
                       for partition in self.partitions]

        self.segment_projections = []

        for split in self.splits:

            max_len = int(np.max(split))
            self.segment_projections.append(nn.Linear(d_model,max_len))

        self.segment_projections = nn.ModuleList(self.segment_projections)

    def forward(self,x):
        """
        x: [batch,channel, leftmost tokens, latent]
        output: [batch, time = f, channel]
        """

        outs = []

        for i in range(len(self.partitions)):

            x_r = x[...,self.parition_indices[i]:self.parition_indices[i+1],:]
            # > [batch,channel - series,latent channel - specific patches corresponding to a resolution, hidden]

            projected_r = self.segment_projections[i](x_r)
            # > [batch,channel - series,latent channel - specific patches corresponding to a resolution, max_len for resolution]

            masked_r = [projected_r[...,s,:self.splits[i][s]] for s in range(self.partitions[i])]
            # > [[batch,channel - series,split] for each split]
            # mask out unnecessary projections in case the partition does not divide the forecast horizon

            resolution_forecast = torch.concat(masked_r,dim=-1)
            # > [batch,channel - series, time - forecast horizon]
            #concatenate across the segments in the resolution

            outs.append(resolution_forecast)

        outs = torch.stack(outs, dim = -1)
        # > [batch,channel - series, time - forecast horizon, partition/resolution]

        out = torch.sum(outs,dim = -1)
        # > [batch,channel - series, time - forecast horizon]

        out = out.transpose(-1,-2)
        # > [batch, time, channel]

        return out

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, n_tokens, learn_pe = True):
        super(LearnablePositionalEncoding, self).__init__()

        LPE = torch.empty((n_tokens, d_model))
        nn.init.uniform_(LPE, -0.02, 0.02)

        self.LPE = nn.Parameter(LPE, requires_grad=learn_pe)

    def forward(self,x):
        return x + self.LPE


class AuxiliaryEncoding(nn.Module):
    """
    This module processes all auxiliary encodings which are time varying.
    This is achieved using numerical and categorical embeddings and a basis splitter module.
    """
    def __init__(self, d_model, lookback_horizon,
                forecast_horizon, partitions, global_auxiliary_numerical,
                global_auxiliary_categorical_list, densest_partition, squeeze_joint_patch_channels, shared = True):
        super(AuxiliaryEncoding, self).__init__()

        self.shared = shared

        self.squeeze_joint_patch_channels = squeeze_joint_patch_channels

        self.encoding_cat = CategoricalEmbedding(categorical_feature_size_list = global_auxiliary_categorical_list,
                                                d_model = d_model)

        self.encoding_cont = NumericalEmbedding(num_continuous_variables = global_auxiliary_numerical,
                                                d_model = d_model)

        self.feature_mixer = nn.Linear(global_auxiliary_numerical + len(global_auxiliary_categorical_list),1)

        self.splitter_basis = ManySplittersBasisComination(d_model = d_model, horizon = lookback_horizon + forecast_horizon, partitions = partitions)

        if self.squeeze_joint_patch_channels:
            number_partitions = int(np.sum(partitions))
            self.squeeze_joint_patches_layer = nn.Linear(number_partitions,densest_partition)

    def forward(self,x_cont,x_cat):
        """
        input: [batch,time,features], also handles [batch,channel,time,features]
        out: [batch,token,latent]
        """
        encoding_cont = self.encoding_cont(x_cont) # [batch, time, T = number or continuous variables, latent]
        encoding_cat = self.encoding_cat(x_cat) # [batch, time, T =number or categorical variables, latent]

        encoding = torch.cat([encoding_cont,encoding_cat],dim = -2) # [batch, time, T = variables, latent]
        encoding = encoding.transpose(-1,-2) # [batch, time, latent, features]
        mix = self.feature_mixer(encoding).squeeze(dim = -1) # [batch, tokens, latent] features squeezed to just 1

        patch_encoding = self.splitter_basis(mix) # [batch,token,latent]

        if self.squeeze_joint_patch_channels:
            patch_encoding = patch_encoding.transpose(-1,-2) # [batch,latent,patches]
            patch_encoding = self.squeeze_joint_patches_layer(patch_encoding) # [batch,latent,token = densest_partition]
            patch_encoding = patch_encoding.transpose(-1,-2) # [batch,token, latent]

        return patch_encoding

class AuxiliaryEncodingStaitc(nn.Module):
    """
    This module processes all auxiliary encodings which are static and potentially squeezes them to a number of tokens.
    """
    def __init__(self, d_model, static_numerical, static_categorical_list, densest_partition, squeeze_static = False, shared = True):
        super(AuxiliaryEncodingStaitc, self).__init__()

        self.encoding_cat = CategoricalEmbedding(categorical_feature_size_list = static_categorical_list,
                                                d_model = d_model,
                                                )

        self.encoding_cont = NumericalEmbedding(num_continuous_variables = static_numerical,
                                                d_model = d_model)

        self.squeeze = squeeze_static

        if self.squeeze:
            self.squeeze_layer = nn.Linear(static_numerical + len(static_categorical_list),densest_partition)

    def forward(self,s_cont,s_cat):
        """
        input: [batch,channel,features]
        out: [batch,channel,latent]
        """
        encoding_cont = self.encoding_cont(s_cont) # [batch, T = number of continuous variables, latent]
        encoding_cat = self.encoding_cat(s_cat) # [batch,T = number of categorical variables, latent]

        encoding = torch.cat([encoding_cont,encoding_cat],dim = -2) #[batch,T = number of static vars, latent]

        if self.squeeze:
            encoding = encoding.transpose(-1,-2) # [batch,latent,T]
            encoding = self.squeeze_layer(encoding) # [batch,latent, T = densest_partition]
            encoding = encoding.transpose(-1,-2) # [batch,T = densest_partition,latent]
           

        return encoding
    
class LightningMultipleResolutionTokenization(pl.LightningModule):
    """
    Construct the model as a lightning module which can handle training and scaling.

    inputs:
        d_model - latent dimension of the model
        d_ff - latent dimension of the feedforward network in encoder
        d_mixer - latent dimension of the compression in the mixer module
        n_head - number of heads in multi-head attention (should divide d_model)
        N - number of encoder stacks
        n_out - number of channels (only used for regularisation)
        densest_partition - number of tokens auxiliraies are compressed into, if set_partitions not defined then they will be set to list(range(densest_partition))
        dropout
        ablation_options - defined in run_pricing.py argument parser
        norm = "BatchNorm"
        activation = "ReLU"
        revin - use reversible instance normalisation bool
        formatter - data formatting class, holds information about what the architecture should be set as
        set_partitions - resolutions of the multiple resolution tokenization
      
        loss = None
        logging_metrics = []
        optimizer_torch = None
        learning_rate = 3e-4
    """
    def __init__(self,
                 d_model,
                 d_ff,
                 d_mixer,
                 n_head,
                 N,
                 n_out,
                 densest_partition,
                 dropout = 0.0,
                 ablation_options = [],
                 norm = "BatchNorm",
                 activation = "GeLU",
                 revin = True,
                 formatter = None,
                 set_partitions = [],

                 loss = None,
                 logging_metrics = [],
                 optimizer_torch = None,
                 learning_rate = 3e-4,
                 **kwargs,
                ):
        super().__init__(**kwargs)

        assert loss, "A loss function must be defined"
        assert optimizer_torch, "A torch optimizer must be defined"
        assert densest_partition > 0 and type(densest_partition) == int, "The number of patches must be a positive integer"


        # Model capabilities/active assets
        self.ablation_options = ablation_options
        self.revin = revin

        if 'shared_weighs' in self.ablation_options:
            self.shared = True
        else:
            self.shared = False

        if 'auxiliary' in self.ablation_options:
            self.auxiliary = True
        else:
            self.auxiliary = False
            
        if 'squeeze_static' in self.ablation_options:
            self.squeeze_static = True
        else:
            self.squeeze_static = False
        
        if 'squeeze_known_temporal_variables' in self.ablation_options:
            self.squeeze_joint_patch_channels = True
        else:
            self.squeeze_joint_patch_channels = False

        if 'channel_mixer' in self.ablation_options:
            self.channel_mixer = True
        else:
            self.channel_mixer = False

        if 'PE' in self.ablation_options:
            self.PE = True
        else:
            self.PE = False


        # Model specifics
        self.formatter = formatter

        forecast_horizon = self.formatter._params["forecast_horizon"]
        lookback_horizon = self.formatter._params["lookback_horizon"]

        self.sizes_dct = get_counts_multi(self.formatter)
        self.n_channels = self.sizes_dct["target"]

        
        
        # Training
        self.logging_metrics = logging_metrics
        self.loss_fn = loss
        self.optimizer_torch = optimizer_torch
        self.learning_rate = learning_rate

        # Model
        if self.revin:
            subtract_last = True
            affine= True
            self.inorm = RevIN(n_out,subtract_last = subtract_last, affine = affine)

        # Patch/Resolution determination
        self.n_splitter_embeddings = int(((densest_partition+1)*densest_partition)/2)
        set_partitions = [int(i) for i in set_partitions]

        if set_partitions:
            self.n_splitter_embeddings = int(np.sum(set_partitions))
            self.partitions = set_partitions
        else:
            self.n_splitter_embeddings = int(((densest_partition+1)*densest_partition)/2)
            self.partitions = list(range(1,densest_partition+1))


        self.splitters = ManySplitters(d_model = d_model,lookback_horizon = lookback_horizon,partitions = self.partitions,
                                        shared = self.shared)
        self.encoder = AdjustedEncoder(d_model = d_model,d_ff = d_ff,n_head = n_head, N = N, dropout = dropout,norm = norm, activation = activation)

        self.output_layer = OutputLayerReverseSegments(d_model = d_model,
                                                        partitions = self.partitions,
                                                        forecast_horizon = forecast_horizon)
        if self.auxiliary:

            if self.sizes_dct["x_continuous_global"] + len(self.sizes_dct["x_categorical_global"]):

                self.global_auxiliary_encoding = AuxiliaryEncoding(d_model = d_model,
                                                                        lookback_horizon = lookback_horizon,
                                                                        forecast_horizon = forecast_horizon,
                                                                        partitions = self.partitions,
                                                                        global_auxiliary_numerical = self.sizes_dct["x_continuous_global"],
                                                                        global_auxiliary_categorical_list = self.sizes_dct["x_categorical_global"],
                                                                        densest_partition = densest_partition,
                                                                        squeeze_joint_patch_channels = self.squeeze_joint_patch_channels,
                                                                        )
                if self.squeeze_joint_patch_channels:
                    self.n_splitter_embeddings += densest_partition
                else:
                    self.n_splitter_embeddings += int(np.sum(set_partitions))


            if self.sizes_dct["x_continuous_specific"] + len(self.sizes_dct["x_categorical_specific"]):

                self.specific_auxuliary_encoding = AuxiliaryEncoding(d_model = d_model,
                                                                        lookback_horizon = lookback_horizon,
                                                                        forecast_horizon = forecast_horizon,
                                                                        partitions = self.partitions,
                                                                        global_auxiliary_numerical = self.sizes_dct["x_continuous_specific"],
                                                                        global_auxiliary_categorical_list = self.sizes_dct["x_categorical_specific"],
                                                                        densest_partition = densest_partition,
                                                                        squeeze_joint_patch_channels = self.squeeze_joint_patch_channels,
                                                                        )

                if self.squeeze_static:
                    self.n_splitter_embeddings += densest_partition
                else:
                    self.n_splitter_embeddings += int(np.sum(set_partitions))

            if self.sizes_dct["s_continuous_global"] + len(self.sizes_dct["s_categorical_global"]):

                self.global_static_auxuliary_encoding = AuxiliaryEncodingStaitc(d_model = d_model,
                                                                        static_numerical = self.sizes_dct["s_continuous_global"],
                                                                        static_categorical_list = self.sizes_dct["s_categorical_global"],
                                                                        densest_partition = densest_partition,
                                                                        squeeze_static = self.squeeze_static,
                                                                        )

                if self.squeeze_static:
                    self.n_splitter_embeddings += densest_partition
                else:
                    self.n_splitter_embeddings += self.sizes_dct["s_continuous_global"] + len(self.sizes_dct["s_categorical_global"])

            if self.sizes_dct["s_continuous_specific"] + len(self.sizes_dct["s_categorical_specific"]):

                self.specific_static_auxuliary_encoding = AuxiliaryEncodingStaitc(d_model = d_model,
                                                                        static_numerical = self.sizes_dct["s_continuous_specific"],
                                                                        static_categorical_list = self.sizes_dct["s_categorical_specific"],
                                                                        densest_partition = densest_partition,
                                                                        squeeze_static = self.squeeze_static,
                                                                        )

                if self.squeeze_joint_patch_channels:
                    self.n_splitter_embeddings += densest_partition
                else:
                    self.n_splitter_embeddings += self.sizes_dct["s_continuous_specific"] + len(self.sizes_dct["s_categorical_specific"])

        if self.channel_mixer:
            self.channel_mixer = AdjustedMixer(n_channels = self.n_channels, n_splitter_embeddings = self.n_splitter_embeddings,
                                        d_model = d_mixer,norm = norm,dropout = dropout,activation = activation, densest_partition = densest_partition,
                                        squeeze_joint_patch_channels = self.squeeze_joint_patch_channels)
            if self.squeeze_joint_patch_channels:
                self.n_splitter_embeddings += densest_partition
            else:
                self.n_splitter_embeddings += int(np.sum(set_partitions))


        if self.PE:
            self.LPE = LearnablePositionalEncoding(d_model = d_model, n_tokens = self.n_splitter_embeddings)



        self.save_hyperparameters(ignore=['loss','logging_metrics'])



    def forward(self, data: torch.Tensor) -> torch.Tensor:
        s_cont_specific = data["s_continuous_specific"]
        s_cat_specific = data["s_categorical_specific"]

        s_cont_global = data["s_continuous_global"]
        s_cat_global = data["s_categorical_global"]

        y_others_continuous = data["y_others_continuous_past"]

        x_cont_past = data["x_continuous_global_past"]
        x_cont_future = data["x_continuous_global_future"]
        x_cont = torch.cat([x_cont_past,x_cont_future],dim=-2)


        x_cat_past = data["x_categorical_global_past"]
        x_cat_future = data["x_categorical_global_future"]
        x_cat = torch.cat([x_cat_past,x_cat_future],dim=-2)

        x_cont_past_specific = data["x_continuous_specific_past"]
        x_cont_future_specific = data["x_continuous_specific_future"]
        x_cont_specific = torch.cat([x_cont_past_specific,x_cont_future_specific],dim=-2)

        x_cat_past_specific = data["x_categorical_specific_past"]
        x_cat_future_specific = data["x_categorical_specific_future"]
        x_cat_specific = torch.cat([x_cat_past_specific,x_cat_future_specific],dim=-2)


        target_past = data["target_past"] # [batch, time, channel]

        if self.revin:
            target_past = self.inorm(target_past,"norm")
        

        inp = Transpose(-1,-2)(target_past)  # [batch, channel, time]
        
        inp = self.splitters(inp)  # [batch, channel, token, latent]

        if self.auxiliary:
            if self.sizes_dct["x_continuous_global"] + len(self.sizes_dct["x_categorical_global"]):
                auxiliary_global = self.global_auxiliary_encoding(x_cont = x_cont, x_cat = x_cat).unsqueeze(1) # [batch, 1 , patch, hidden]
                auxiliary_global = auxiliary_global.repeat(1,self.n_channels,1,1) # repeat same value across channels
                # [batch, channel, patch, hidden]
                inp = torch.cat([inp,auxiliary_global],dim = -2)

            if self.sizes_dct["x_continuous_specific"] + len(self.sizes_dct["x_categorical_specific"]):
                auxiliary_specific = self.specific_auxuliary_encoding(x_cont = x_cont_specific, x_cat = x_cat_specific)
                # [batch, channel, patch, hidden]
                inp = torch.cat([inp,auxiliary_specific],dim = -2)

            if self.sizes_dct["s_continuous_global"] + len(self.sizes_dct["s_categorical_global"]):
                auxiliary_global_static = self.global_static_auxuliary_encoding(s_cont = s_cont_global, s_cat = s_cat_global).unsqueeze(1) # [batch, 1 , patch, hidden]
                auxiliary_global_static = auxiliary_global_static.repeat(1,self.n_channels,1,1) # repeat same value across channels
                # [batch, channel, patch, hidden]
                inp = torch.cat([inp,auxiliary_global_static],dim = -2)

            if self.sizes_dct["s_continuous_specific"] + len(self.sizes_dct["s_categorical_specific"]):

                auxiliary_specific_static = self.specific_static_auxuliary_encoding(s_cont = s_cont_specific, s_cat = s_cat_specific)
                inp = torch.cat([inp,auxiliary_specific_static],dim = -2)

        if self.channel_mixer:
            inp2 = self.channel_mixer(inp)
            inp2 = inp2.repeat(1,self.n_channels,1,1) # repeat same value across channels
            inp = torch.cat([inp,inp2],dim = -2)

        if self.PE:
            inp = self.LPE(inp)

        inp = self.encoder(inp)[-1]
        pred = self.output_layer(inp)
        if self.revin:
            pred = self.inorm(pred,"denorm")

        return pred


    def _calculate_loss(self, batch, loss_fn, mode="train"):
        inp = batch
        target = batch["target_future"]

        output = self(inp)
        loss = 0
        num_series = target.size()[-1]

        for i in range(num_series):
            loss += loss_fn(output[:,:,i], target[:,:,i])
        loss /= num_series

        return loss

    def _calculate_loss_and_metrics(self, batch, loss_fn, mode="val",rescale = True):

        inp = batch
        target = batch["target_future"]
        output = self(inp)
        
        if rescale:
            groups = batch['group_id']
            target = self.rescale(target, groups)
            output = self.rescale(output, groups)
        
        loss = 0
        num_series = target.size()[-1]
        for i in range(num_series):
            loss += loss_fn(output[:,:,i], target[:,:,i])
        loss /= num_series


        other_metric_losses = []
        for metric in self.logging_metrics:
            metric_loss = 0
            for i in range(num_series):
                metric_loss += metric(output[:,:,i], target[:,:,i])
            metric_loss /= num_series

            other_metric_losses.append(metric_loss)

        return loss, other_metric_losses

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        mode = "train"
        loss = self._calculate_loss(batch, loss_fn = self.loss_fn, mode = mode)
        self.log("%s_loss" % mode, loss.detach(), batch_size=batch['target_past'].size()[0])

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        mode = "val"
        with torch.no_grad():
            loss, other_metric_losses = self._calculate_loss_and_metrics(batch,loss_fn = self.loss_fn, mode=mode)
        self.log("%s_loss" % mode, loss, batch_size=batch['target_past'].size()[0])

        for i, metric in enumerate(self.logging_metrics):
            name = str(type(metric)).split("'>")[0].split('.')[-1]
            self.log(f"{mode}_loss_{name}", other_metric_losses[i], batch_size=batch['target_past'].size()[0])



    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        mode = "test"
        with torch.no_grad():
            loss, other_metric_losses = self._calculate_loss_and_metrics(batch,loss_fn = self.loss_fn, mode=mode)
        self.log("%s_loss" % mode, loss, batch_size=batch['target_past'].size()[0])

        for i, metric in enumerate(self.logging_metrics):
            name = str(type(metric)).split("'>")[0].split('.')[-1]
            self.log(f"{mode}_loss_{name}", other_metric_losses[i], batch_size=batch['target_past'].size()[0])

    def rescale(self, pred, groups):
        
        rescaled = []
        for i, group in enumerate(groups):
            rescaled.append(torch.tensor(self.formatter._target_scaler[group].inverse_transform(pred.detach().numpy()[i])).unsqueeze(0))
        
        rescaled = torch.concat(rescaled,dim = 0)

        return rescaled
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0, rescale = True):

        
        groups = batch['group_id']
        sequence_ids = batch['sequence_id']
        self.eval()
        with torch.no_grad():
            pred = self(batch)
        self.train()
        
        if rescale:
            pred = self.rescale(pred, groups)
        
        return pred, sequence_ids
        

    def configure_optimizers(self):
        optimizer = self.optimizer_torch(self.parameters(),lr = self.learning_rate)
        return optimizer
