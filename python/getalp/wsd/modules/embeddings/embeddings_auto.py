import torch
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from getalp.wsd.torch_utils import default_device
from typing import List, Union, Dict


class EmbeddingsAuto(Module):

    def __init__(self, auto_model: str, auto_path: str):
        super().__init__()
        if "camembert" in auto_model:
            from transformers import CamembertModel, CamembertTokenizer
            self.auto_embeddings = CamembertModel.from_pretrained(auto_path)
            self.auto_tokenizer = CamembertTokenizer.from_pretrained(auto_path)
        elif "flaubert" in auto_model:
            from transformers import XLMModel, XLMTokenizer
            self.auto_embeddings = XLMModel.from_pretrained(auto_path)
            self.auto_tokenizer = XLMTokenizer.from_pretrained(auto_path)
            self.auto_tokenizer.do_lowercase_and_remove_accent = False
        elif "xlm" in auto_model:
            from transformers import XLMModel, XLMTokenizer
            self.auto_embeddings = XLMModel.from_pretrained(auto_path)
            self.auto_tokenizer = XLMTokenizer.from_pretrained(auto_path)
        elif "roberta" in auto_model:
            from transformers import RobertaModel, RobertaTokenizer
            self.auto_embeddings = RobertaModel.from_pretrained(auto_path)
            self.auto_tokenizer = RobertaTokenizer.from_pretrained(auto_path)
        elif "bert" in auto_model:
            from transformers import BertModel, BertTokenizer
            self.auto_embeddings = BertModel.from_pretrained(auto_path)
            self.auto_tokenizer = BertTokenizer.from_pretrained(auto_path)
        else:
            from transformers import AutoModel, AutoTokenizer, XLMTokenizer
            self.auto_embeddings = AutoModel.from_pretrained(auto_path)
            self.auto_tokenizer = AutoTokenizer.from_pretrained(auto_path)
            if isinstance(self.auto_tokenizer, XLMTokenizer):
                self.auto_tokenizer.do_lowercase_and_remove_accent = False
        for param in self.auto_embeddings.parameters():
            param.requires_grad = False
        self._is_fixed = True
        self._output_dim = self.auto_embeddings.config.hidden_size
        self._begin_special_token_count = self.get_begin_special_token_count()
        self._padding_id = self.auto_tokenizer.pad_token_id

    # input:
    #   - sample_x:  List[str]  - seq_in
    # output:
    #   - sample_x:  LongTensor - seq_out
    #   - new_size:  int        - seq_out
    #   - indices:   List[int]  - seq_in
    def preprocess_sample_first(self, sample_x):
        seq_token_indices: List[int] = []
        seq_tokens: Union[List[str], torch.Tensor] = []
        current_index = self._begin_special_token_count
        for token in sample_x:
            subtokens = self.auto_tokenizer.tokenize(token)
            seq_token_indices.append(current_index)
            current_index += len(subtokens)
            for subtoken in subtokens:
                seq_tokens.append(subtoken)
        seq_tokens = self.auto_tokenizer.convert_tokens_to_ids(seq_tokens)
        seq_tokens = self.auto_tokenizer.build_inputs_with_special_tokens(seq_tokens)
        seq_tokens = torch.tensor(seq_tokens, dtype=torch.long)
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
        pad_mask = [torch.ones_like(x) for x in inputs]
        pad_mask = pad_sequence(pad_mask, batch_first=True, padding_value=0)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self._padding_id)
        inputs = self.auto_embeddings(inputs, attention_mask=pad_mask)
        return inputs[0], pad_mask, token_indices

    def get_begin_special_token_count(self):
        from transformers import BertTokenizer, XLMTokenizer
        if isinstance(self.auto_tokenizer, BertTokenizer):
            return 1
        if isinstance(self.auto_tokenizer, XLMTokenizer):
            return 1
        else:
            from transformers import CamembertTokenizer
            if isinstance(self.auto_tokenizer, CamembertTokenizer):
                return 1
            else:
                raise NotImplementedError

    def get_output_dim(self):
        return self._output_dim

    def is_fixed(self):
        return self._is_fixed

    def get_lut_embeddings(self):
        return self.auto_embeddings.word_embeddings


_auto_embeddings_wrapper: Dict[str, EmbeddingsAuto] = {}


def get_auto_embeddings(auto_model: str, auto_path: str, clear_text: bool):
    assert clear_text
    if auto_path not in _auto_embeddings_wrapper:
        _auto_embeddings_wrapper[auto_path] = EmbeddingsAuto(auto_model, auto_path)
    return _auto_embeddings_wrapper[auto_path]
