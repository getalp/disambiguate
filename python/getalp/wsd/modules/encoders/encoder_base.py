from torch.nn import Module, Dropout, Linear, ModuleList
from getalp.wsd.torch_fix import torch_cat
from getalp.wsd.model_config import ModelConfig


class EncoderBase(Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.input_resize = None
        for i in range(config.data_config.input_features):
            if config.input_resize[i] is not None and self.input_resize is None:
                self.input_resize = ModuleList()

        if self.input_resize is not None:
            self.resulting_embeddings_size = 0
            for i in range(config.data_config.input_features):
                if config.input_resize[i] is not None:
                    self.input_resize.append(Linear(in_features=config.input_embeddings_sizes[i], out_features=config.input_resize[i]))
                    self.resulting_embeddings_size += config.input_resize[i]
                else:
                    self.input_resize.append(None)
                    self.resulting_embeddings_size += config.input_embeddings_sizes[i]
        else:
            self.resulting_embeddings_size = sum(config.input_embeddings_sizes)

        if config.input_apply_linear:
            if config.input_linear_size is None:
                self.input_linear = Linear(in_features=self.resulting_embeddings_size, out_features=self.resulting_embeddings_size)
            else:
                self.input_linear = Linear(in_features=self.resulting_embeddings_size, out_features=config.input_linear_size)
                self.resulting_embeddings_size = config.input_linear_size
        else:
            self.input_linear = None

        if config.input_dropout_rate is not None:
            self.input_dropout = Dropout(p=config.input_dropout_rate)
        else:
            self.input_dropout = None

        config.encoder_output_size = self.resulting_embeddings_size

    # input:
    #   - embeddings: List[FloatTensor] - features x batch x seq x hidden
    #   - pad_mask:   LongTensor        - batch x seq
    # output:
    #   - output      FloatTensor       - batch x seq x hidden
    def forward(self, embeddings, pad_mask):
        if self.input_resize is not None:
            for i in range(len(embeddings)):
                if self.input_resize[i] is not None:
                    embeddings[i] = self.input_resize[i](embeddings[i])
        embeddings = torch_cat(embeddings, dim=2)
        if self.input_linear is not None:
            embeddings = self.input_linear(embeddings)
        if self.input_dropout is not None:
            embeddings = self.input_dropout(embeddings)
        return embeddings
