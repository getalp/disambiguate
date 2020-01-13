from torch.nn import Module, LayerNorm, ModuleList, Dropout
from getalp.wsd.model_config import ModelConfig
from onmt.encoders.transformer import TransformerEncoderLayer
from getalp.wsd.modules.encoders.encoder_base import EncoderBase
from getalp.wsd.common import pad_token_index
from getalp.wsd.modules import PositionalEncoding
import math


class EncoderTransformer(Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.base = EncoderBase(config)

        if config.encoder_transformer_positional_encoding:
            self.positional_encoding = PositionalEncoding(self.base.resulting_embeddings_size)
            # self.add_module("pe", self.positional_encoding)
        else:
            self.positional_encoding = None

        if config.encoder_transformer_scale_embeddings:
            self.embeddings_scale = math.sqrt(float(self.base.resulting_embeddings_size))
        else:
            self.embeddings_scale = None

        self.dropout = Dropout(config.encoder_transformer_dropout)

        self.transformer = ModuleList([TransformerEncoderLayer(d_model=self.base.resulting_embeddings_size,
                                                               heads=config.encoder_transformer_heads,
                                                               d_ff=config.encoder_transformer_hidden_size,
                                                               dropout=config.encoder_transformer_dropout,
                                                               attention_dropout=config.encoder_transformer_dropout)
                                       for _ in range(config.encoder_transformer_layers)])
        self.layer_norm = LayerNorm(self.base.resulting_embeddings_size, eps=1e-6)

        config.encoder_output_size = self.base.resulting_embeddings_size

    # input:
    #   - embeddings     List[FloatTensor] - features x batch x seq x hidden
    #   - pad_mask       LongTensor        - batch x seq
    # output:
    #   - output         FloatTensor       - batch x seq x hidden
    def forward(self, embeddings, pad_mask):
        embeddings = self.base(embeddings, pad_mask)  # batch x seq x hidden
        if self.embeddings_scale is not None:
            embeddings = embeddings * self.embeddings_scale
        if self.positional_encoding is not None:
            embeddings = embeddings + self.positional_encoding(embeddings.size(1))
        embeddings = self.dropout(embeddings)
        pad_mask = pad_mask.eq(pad_token_index).unsqueeze(1)  # batch x 1 x seq
        for layer in self.transformer:
            embeddings = layer(embeddings, pad_mask)
        embeddings = self.layer_norm(embeddings)
        return embeddings
