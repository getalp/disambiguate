from torch.nn import Module, Embedding
from getalp.wsd.common import pad_token_index, unk_token_index
from getalp.wsd.torch_fix import *
from getalp.wsd.torch_utils import default_device


class EmbeddingsLUT(Module):

    def __init__(self, input_embeddings, input_vocabulary_size, input_embeddings_size, clear_text):
        super().__init__()
        assert not clear_text
        if input_embeddings is not None:
            self.lut_embeddings = Embedding.from_pretrained(embeddings=input_embeddings, freeze=True)
            self._is_fixed = True
        else:
            self.lut_embeddings = Embedding(num_embeddings=input_vocabulary_size, embedding_dim=input_embeddings_size, padding_idx=pad_token_index)
            self._is_fixed = False
        self._output_dim = input_embeddings_size

    # input:
    #   - sample_x:  LongTensor  - seq_in
    # output:
    #   - sample_x:  LongTensor  - seq_out
    #   - new_size:  int         - seq_out
    #   - indices:   List[int]   - seq_in
    @staticmethod
    def preprocess_sample_first(sample_x):
        return sample_x, None, None

    # input:
    #   - sample_x:  LongTensor - seq_in
    #   - new_size:  int        - seq_out
    #   - indices:   List[int]  - seq_in
    # output:
    #   - sample_x:  LongTensor - seq_out
    @staticmethod
    def preprocess_sample_next(sample_x, new_size, indices):
        if indices is None:
            return sample_x
        new_sample_x = torch_full([new_size], fill_value=unk_token_index, dtype=torch_long)
        for i in range(len(indices)):
            new_sample_x[indices[i]] = sample_x[i]
        return new_sample_x

    # inputs:
    #   - inputs:        LongTensor      (batch x seq_in)
    # output:
    #   - output:        FloatTensor     (batch x seq_out x hidden)
    #   - pad_mask:      LongTensor      (batch x seq_out)
    #   - token_indices: List[List[int]] (batch x seq_in)
    def forward(self, inputs):
        embeddings = self.lut_embeddings(inputs)
        return embeddings, inputs, None

    def get_output_dim(self):
        return self._output_dim

    def is_fixed(self):
        return self._is_fixed

    def get_lut_embeddings(self):
        return self.lut_embeddings
