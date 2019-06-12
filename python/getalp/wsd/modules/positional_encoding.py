from torch.nn import Module
from getalp.wsd.torch_fix import *
import math


class PositionalEncoding(Module):

    def __init__(self, input_embeddings_size, max_len=5000):
        super().__init__()
        pe = torch_zeros(max_len, input_embeddings_size)  # max_len x input_embeddings_size
        position = torch_arange(start=0, end=max_len, step=1).unsqueeze(1)  # max_len x 1
        div_term = torch_exp((torch_arange(start=0, end=input_embeddings_size, step=2, dtype=torch_float32) * -(math.log(10000.0) / input_embeddings_size)))
        pe[:, 0::2] = torch_sin(position.float() * div_term)
        pe[:, 1::2] = torch_cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe = pe
        self.input_embeddings_size = input_embeddings_size

    # inputs:
    #   - int          (seq)
    # output:
    #   - FloatTensor  (1 x seq x hidden)
    def forward(self, seq: int, full: bool = True):
        if full:
            return self.pe[:, :seq, :]
        else:
            return self.pe[:, seq, :]
