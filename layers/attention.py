import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    '''
    Note: Not adjusted for multiple channels, this is achieved with adjusted modules. 
    However, this is fixable directly within this module and I appologise for not doing it myself due to fear of downstream issues.
    '''
    def __init__(self,d_in, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_attn = d_model // n_head
        self.W_Q = nn.Linear(d_in, d_model)
        self.W_K = nn.Linear(d_in, d_model)
        self.W_V = nn.Linear(d_in, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass through the multi-head attention layer.
        q, k, v: [batch, token, latent]
        mask: [token size, token size)
        out: [batch, token, latent]
        """
        q = self.W_Q(q)
        k = self.W_K(k)
        v = self.W_V(v)

        # Split the query, key, and value tensors into n_head different pieces
        q = q.view(q.size(0), q.size(1), self.n_head, self.d_attn).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.n_head, self.d_attn).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.n_head, self.d_attn).transpose(1, 2)
        # Pass the split tensors to the multi-head attention layer
        attn = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_attn)

        if mask is not None:
            attn = attn.masked_fill(mask == 1, -np.inf)
        attn = F.softmax(attn, dim=-1)
        # Multiply the attention weights with the value tensors
        out = torch.matmul(attn, v)
        # Transpose the result back to (batch_size, seq_len, d_model)
        out = out.transpose(1, 2)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        # Combine the results from the different heads
        out = self.W_O(out)
        return out
