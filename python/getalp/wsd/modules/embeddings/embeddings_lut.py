from torch.nn import Module, Embedding
from getalp.wsd.common import pad_token_index, unk_token_index
from getalp.wsd.torch_fix import *
from torch.nn.utils.rnn import pad_sequence
from getalp.wsd.torch_utils import default_device
from typing import List, Union


class EmbeddingsLUT(Module):

    def __init__(self, input_embeddings: str, input_vocabulary_size: int, input_embeddings_size: int, clear_text: bool, tokenize_model: str):
        super().__init__()
        if clear_text:
            assert tokenize_model is not None
            from pytorch_pretrained_bert import BertTokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained(tokenize_model, do_lower_case=False)
            input_vocabulary_size = len(self.bert_tokenizer.vocab)
            self.lut_embeddings = Embedding(num_embeddings=input_vocabulary_size, embedding_dim=input_embeddings_size, padding_idx=pad_token_index)
            self._is_fixed = False
        else:
            self.bert_tokenizer = None
            if input_embeddings is not None:
                self.lut_embeddings = Embedding.from_pretrained(embeddings=input_embeddings, freeze=True)
                self._is_fixed = True
            else:
                self.lut_embeddings = Embedding(num_embeddings=input_vocabulary_size, embedding_dim=input_embeddings_size, padding_idx=pad_token_index)
                self._is_fixed = False
        self._output_dim = input_embeddings_size

    # input:
    #   - sample_x:  Union[List[str], LongTensor] - seq_in
    # output:
    #   - sample_x:  LongTensor                   - seq_out
    #   - new_size:  int                          - seq_out
    #   - indices:   List[int]                    - seq_in
    def preprocess_sample_first(self, sample_x):
        if self.bert_tokenizer is not None:
            seq_token_indices: List[int] = []
            seq_tokens: Union[List[str], torch.Tensor] = []
            current_index = 1  # 0 is [CLS]
            for token in sample_x:
                subtokens = self.bert_tokenizer.tokenize(token)
                seq_token_indices.append(current_index)
                current_index += len(subtokens)
                for subtoken in subtokens:
                    seq_tokens.append(subtoken)
            seq_tokens = ["[CLS]"] + seq_tokens + ["[SEP]"]
            seq_tokens = self.bert_tokenizer.convert_tokens_to_ids(seq_tokens)
            seq_tokens = torch_tensor(seq_tokens, dtype=torch_long)
            return seq_tokens, seq_tokens.size(0), seq_token_indices
        return sample_x, None, None

    # input:
    #   - sample_x:  LongTensor - seq_in
    #   - new_size:  int        - seq_out
    #   - indices:   List[int]  - seq_in
    # output:
    #   - sample_x:  LongTensor - seq_out
    def preprocess_sample_next(self, sample_x, new_size, indices):
        if self.bert_tokenizer is not None:
            return sample_x, indices
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
        if self.bert_tokenizer is not None:
            tokens: List[torch.Tensor] = []
            token_indices: List[List[int]] = []
            for seq in inputs:
                tokens.append(seq[0].to(default_device))
                token_indices.append(seq[1])
            inputs = tokens
            pad_mask = [torch_ones_like(x) for x in inputs]
            pad_mask = pad_sequence(pad_mask, batch_first=True, padding_value=0)
            inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
            inputs = self.lut_embeddings(inputs)
            return inputs, pad_mask, token_indices
        embeddings = self.lut_embeddings(inputs)
        return embeddings, inputs, None

    def get_output_dim(self):
        return self._output_dim

    def is_fixed(self):
        return self._is_fixed

    def get_lut_embeddings(self):
        return self.lut_embeddings
