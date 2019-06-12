from torch.nn import Module, Linear
from getalp.wsd.model_config import ModelConfig
from getalp.wsd.data_config import DataConfig
from getalp.wsd.torch_fix import *
from getalp.wsd.torch_utils import default_device


class DecoderClassify(Module):

    def __init__(self, config: ModelConfig, data_config: DataConfig):
        super().__init__()

        self.output_linears = []
        for i in range(0, data_config.output_features):
            module = Linear(in_features=config.encoder_output_size, out_features=data_config.output_vocabulary_sizes[i])
            self.output_linears.append(module)
            self.add_module("output_linear" + str(i), module)

    # input:
    #   - inputs:        FloatTensor       - batch x seq_in x hidden
    #   - token_indices: List[List[int]]   - batch x real_seq_in
    # output:
    #   - output:        List[FloatTensor] - batch x real_seq_in x out_vocabulary_dim
    def forward(self, inputs, token_indices):
        if token_indices is not None:
            max_length = max([len(seq) for seq in token_indices])
            new_inputs = torch_zeros(inputs.size(0), max_length, inputs.size(2), dtype=torch_float32, device=default_device)
            for i in range(len(token_indices)):
                for j in range(len(token_indices[i])):
                    new_inputs[i][j] = inputs[i][token_indices[i][j]]
            inputs = new_inputs
        outputs = []
        for i in range(0, len(self.output_linears)):
            outputs.append(self.output_linears[i](inputs))
        return outputs


