from torch.nn import Module
from getalp.wsd.torch_fix import *
from torch.nn.utils.rnn import pad_sequence
from getalp.wsd.torch_utils import default_device
from typing import List, Union, Dict


class EmbeddingsBert(Module):

    def __init__(self, bert_path: str):
        super().__init__()
        from pytorch_pretrained_bert import BertModel, BertTokenizer
        self.bert_embeddings = BertModel.from_pretrained(bert_path)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        for param in self.bert_embeddings.parameters():
            param.requires_grad = False
        self._is_fixed = True
        self._output_dim = self.bert_embeddings.config.hidden_size

    # input:
    #   - sample_x:  List[str]  - seq_in
    # output:
    #   - sample_x:  LongTensor - seq_out
    #   - new_size:  int        - seq_out
    #   - indices:   List[int]  - seq_in
    def preprocess_sample_first(self, sample_x):
        seq_token_indices: List[int] = []
        seq_tokens: Union[List[str], torch.Tensor] = []
        current_index = 1  # 0 is [CLS]
        for token in sample_x:
            subtokens = self.bert_tokenizer.tokenize(token)
            if current_index + len(subtokens) + 1 >= self.bert_tokenizer.max_len:
                break
            seq_token_indices.append(current_index)
            current_index += len(subtokens)
            for subtoken in subtokens:
                seq_tokens.append(subtoken)
        seq_tokens = ["[CLS]"] + seq_tokens + ["[SEP]"]
        seq_tokens = self.bert_tokenizer.convert_tokens_to_ids(seq_tokens)
        seq_tokens = torch_tensor(seq_tokens, dtype=torch_long)
        return seq_tokens, seq_tokens.size(0), seq_token_indices

    # input:
    #   - sample_x:  LongTensor  - seq_in
    #   - new_size:  int         - seq_out
    #   - indices:   List[int]   - seq_in
    # output:
    #   - sample_x:  Tuple[LongTensor, List[int]]  - sample_x, indices
    @staticmethod
    def preprocess_sample_next(sample_x, new_size, indices):
        return sample_x, indices

    # inputs:
    #   - inputs:        List[List[str]]  (batch x seq_in)
    # output:
    #   - output:        FloatTensor      (batch x seq_out x hidden)
    #   - pad_mask:      LongTensor       (batch x seq_out)
    #   - token_indices: List[List[int]]  (batch x seq_in)
    def forward(self, inputs):
        tokens: List[torch.Tensor] = []
        token_indices: List[List[int]] = []
        for seq in inputs:
            tokens.append(seq[0].to(default_device))
            token_indices.append(seq[1])
        inputs = tokens
        pad_mask = [torch_ones_like(x) for x in inputs]
        pad_mask = pad_sequence(pad_mask, batch_first=True, padding_value=0)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        inputs, _ = self.bert_embeddings(inputs, attention_mask=pad_mask, output_all_encoded_layers=False)
        return inputs, pad_mask, token_indices

    def get_output_dim(self):
        return self._output_dim

    def is_fixed(self):
        return self._is_fixed

    def get_lut_embeddings(self):
        return self.bert_embeddings.embeddings.word_embeddings


_bert_embeddings_wrapper: Dict[str, EmbeddingsBert] = {}


def get_bert_embeddings(bert_path: str, clear_text: bool):
    assert clear_text
    if bert_path not in _bert_embeddings_wrapper:
        _bert_embeddings_wrapper[bert_path] = EmbeddingsBert(bert_path)
    return _bert_embeddings_wrapper[bert_path]
