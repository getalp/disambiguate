from torch.nn import Module
import torch
from getalp.wsd.model_config import ModelConfig
from getalp.wsd.data_config import DataConfig


class DecoderTranslateTransformer(Module):

    def __init__(self, config: ModelConfig, data_config: DataConfig, encoder_embeddings):
        super().__init__()
        raise NotImplementedError

    def forward(self, encoder_output: torch.Tensor, pad_mask: torch.Tensor, true_output: torch.Tensor):
        raise NotImplementedError
