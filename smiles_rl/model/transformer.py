import torch.nn as tnn

from ..utils.layers import *

class Transformer(tnn.Module):
    def __init__(self, voc_size: int, max_seq_len: int=1000, layer_size: int=256, n_layers: int=6, n_heads: int=8, dropout: float=0.0):
        super().__init__()   
        self.layer_size = layer_size     
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.decoder = DecoderOnly(voc_size, self.layer_size, max_seq_len, n_layers, n_heads, dropout)
        self.out = tnn.Linear(self.layer_size, voc_size)

    def forward(self, trg, trg_mask=None):
        d_output = self.decoder(trg, trg_mask)
        output = self.out(d_output)
        return output
    
    def get_params(self):
        return {
            "layer_size": self.layer_size,
            "num_layers": self.n_layers,
            "num_heads" : self.n_heads,
            "dropout": self.dropout,
        }
    
    def freeze_layers(self, layers_to_freeze):
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False