import copy

import math

import numpy as np

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.nn.init as init

def get_clones(module, N):
    return tnn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Embedder(tnn.Module):
    """
    input: shape(batch_size, seq_len, voc_size)
    output: shape(batch_size, seq_len, layer_size)
    """
    def __init__(self, voc_size, layer_size):
        super(Embedder, self).__init__()
        self.layer_size = layer_size
        self.embed = tnn.Embedding(voc_size, layer_size)
        self._reset()
    def forward(self, x):
        return self.embed(x)
    def _reset(self,):
        tnn.init.normal_(self.embed.weight, mean=0, std=np.sqrt(1 / self.layer_size))

class PositionalEncoder(tnn.Module):
    def __init__(self, layer_size, max_seq_len, dropout):
        super(PositionalEncoder, self).__init__()
        self.layer_size = layer_size
        self.dropout = tnn.Dropout(dropout)
        self.div_term = torch.exp(torch.arange(0, layer_size, 2).float() * -(math.log(10000.0) / layer_size))

        self.pe = self.create_positional_encoding(max_seq_len, layer_size)

    def create_positional_encoding(self, length, layer_size):
        position = torch.arange(length).unsqueeze(1).float()
        pe = torch.zeros(length, layer_size)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x * math.sqrt(self.layer_size)
        seq_len = x.size(1)
        if seq_len > self.pe.size(2):
            self.pe = self.create_positional_encoding(seq_len, self.layer_size).to(x.device)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
    
class Norm(tnn.Module):
    """
    input: shape(batch_size, seq_len, layer_size)
    output: shape(batch_size, seq_len, layer_size)
    """
    def __init__(self, layer_size, eps = 1e-6):
        super().__init__()
        self.size = layer_size
        # create two learnable parameters to calibrate normalisation
        self.alpha = tnn.Parameter(torch.ones(self.size))
        self.bias  = tnn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e10)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(tnn.Module):
    """
    input: shape(batch_size, seq_len, layer_size)
    q: shape(batch_size, (n_heads, split_seq_len), layer_size)
    v: shape(batch_size, (n_heads, split_seq_len), layer_size)
    k: shape(batch_size, (n_heads, split_seq_len), layer_size)
    output: shape(batch_size, seq_len, layer_size)
    """
    def __init__(self, heads, layer_size, dropout):
        super().__init__()
        self.layer_size = layer_size
        self.d_k = layer_size // heads
        self.h = heads
        self.q_linear = tnn.Linear(layer_size, layer_size)
        self.v_linear = tnn.Linear(layer_size, layer_size)
        self.k_linear = tnn.Linear(layer_size, layer_size)
        self.dropout = tnn.Dropout(dropout)
        self.out = tnn.Linear(layer_size, layer_size)     
        self._reset()

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * N * sl * layer_size
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.layer_size)
        output = self.out(concat)
        return output
    
    def _reset(self):
        init.xavier_uniform_(self.q_linear.weight)
        init.xavier_uniform_(self.k_linear.weight)
        init.xavier_uniform_(self.v_linear.weight)
        init.xavier_uniform_(self.out.weight)

        init.constant_(self.q_linear.bias, 0)
        init.constant_(self.k_linear.bias, 0)
        init.constant_(self.v_linear.bias, 0)
        init.constant_(self.out.bias, 0)

class FeedForward(tnn.Module):
    def __init__(self, layer_size, dropout, d_ff=2048):
        super().__init__()
        self.linear_1 = tnn.Linear(layer_size, d_ff)
        self.dropout = tnn.Dropout(dropout)
        self.linear_2 = tnn.Linear(d_ff, layer_size)
        self._reset()
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    def _reset(self,):
        tnn.init.xavier_uniform_(self.linear_1.weight)
        tnn.init.xavier_uniform_(self.linear_2.weight)

        if self.linear_1.bias is not None:
            tnn.init.constant_(self.linear_1.bias, 0)
        if self.linear_2.bias is not None:
            tnn.init.constant_(self.linear_2.bias, 0)


class EncoderLayer(tnn.Module):
    def __init__(self, layer_size, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(layer_size)
        self.norm_2 = Norm(layer_size)
        self.attn   = MultiHeadAttention(heads, layer_size, dropout=dropout)
        self.ff     = FeedForward(layer_size, dropout=dropout)
        self.dropout_1 = tnn.Dropout(dropout)
        self.dropout_2 = tnn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask)) # self attention only
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(tnn.Module):
    def __init__(self, layer_size, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(layer_size)
        self.norm_2 = Norm(layer_size)
        self.norm_3 = Norm(layer_size)

        self.dropout_1 = tnn.Dropout(dropout)
        self.dropout_2 = tnn.Dropout(dropout)
        self.dropout_3 = tnn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, layer_size, dropout)
        self.attn_2 = MultiHeadAttention(heads, layer_size, dropout)
        self.ff = FeedForward(layer_size, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x )
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask)) 
        x2 = self.norm_3(x) 
        x = x + self.dropout_3(self.ff(x2))
        return x

class Encoder(tnn.Module):
    def __init__(self, voc_size, layer_size, max_seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed  = Embedder(voc_size, layer_size)
        self.pe     = PositionalEncoder(layer_size, max_seq_len, dropout)
        self.layers = get_clones(EncoderLayer(layer_size, heads, dropout), N)
        self.norm   = Norm(layer_size)

    def forward(self,
                src,
                mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(tnn.Module):
    def __init__(self, voc_size, layer_size, max_seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed   = Embedder(voc_size, layer_size)
        self.pe      = PositionalEncoder(layer_size, max_seq_len, dropout)
        self.layers  = get_clones(DecoderLayer(layer_size, heads, dropout), N)
        self.norm    = Norm(layer_size)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
    
class DecoderLayerOnly(tnn.Module):
    def __init__(self, layer_size, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(layer_size)
        self.norm_2 = Norm(layer_size)

        self.dropout_1 = tnn.Dropout(dropout)
        self.dropout_2 = tnn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, layer_size, dropout)
        self.ff = FeedForward(layer_size, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2)) 
        return x 

class DecoderOnly(tnn.Module):
    def __init__(self, voc_size, layer_size, max_seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed   = Embedder(voc_size, layer_size)
        self.pe      = PositionalEncoder(layer_size, max_seq_len, dropout)
        self.layers  = get_clones(DecoderLayerOnly(layer_size, heads, dropout), N)
        self.norm    = Norm(layer_size)

    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)