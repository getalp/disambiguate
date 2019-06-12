from torch.nn import Module
from typing import List, Dict, Tuple
from getalp.wsd.torch_utils import default_device


class EmbeddingsElmo(Module):

    # elmo_path = {small, medium, original}
    def __init__(self, elmo_path: str, input_vocabulary: List[str], clear_text: bool):
        super().__init__()
        from allennlp.modules.elmo import Elmo
        if elmo_path in _elmo_models_map:
            options_file_path, weights_file_path = _elmo_models_map[elmo_path]
        else:
            options_file_path, weights_file_path = elmo_path + "_options.json", elmo_path + "_weights.hdf5"
        self.elmo_embeddings = Elmo(options_file=options_file_path, weight_file=weights_file_path, num_output_representations=1, vocab_to_cache=input_vocabulary)
        self.clear_text = clear_text

    # input:
    #   - sample_x:  Union[List[str], LongTensor]  - seq_in
    # output:
    #   - sample_x:  Union[List[str], LongTensor]  - seq_out
    #   - new_size:  int                           - seq_out
    #   - indices:   List[int]                     - seq_in
    @staticmethod
    def preprocess_sample_first(sample_x):
        return sample_x, None, None

    # input:
    #   - sample_x:  Union[List[str], LongTensor]  - seq_in
    #   - new_size:  int                           - seq_out
    #   - indices:   List[int]                     - seq_in
    # output:
    #   - sample_x:  Union[List[str], LongTensor]  - seq_out
    @staticmethod
    def preprocess_sample_next(sample_x, new_size, indices):
        return sample_x

    # inputs:
    #   - inputs:        Union[List[List[str]], LongTensor]  (batch x seq_in)
    # output:
    #   - output:        FloatTensor                         (batch x seq_out x hidden)
    #   - pad_mask:      LongTensor                          (batch x seq_out)
    #   - token_indices: List[List[int]]                     (batch x seq_in)
    def forward(self, inputs):
        if self.clear_text:
            from allennlp.modules.elmo import batch_to_ids
            inputs = batch_to_ids(inputs)
            inputs = inputs.to(default_device)
            return self.elmo_embeddings(inputs)["elmo_representations"][0], None, None
        else:
            return self.elmo_embeddings(inputs, inputs)["elmo_representations"][0], inputs, None

    def get_output_dim(self):
        return self.elmo_embeddings.get_output_dim()

    @staticmethod
    def is_fixed():
        return True


_elmo_embeddings_wrapper: Dict[Tuple[str, Tuple[str], bool], EmbeddingsElmo] = {}


def get_elmo_embeddings(elmo_path: str, input_vocabulary: List[str], clear_text: bool):
    hashable_parameters = (elmo_path, tuple(input_vocabulary), clear_text)
    if elmo_path not in _elmo_embeddings_wrapper:
        _elmo_embeddings_wrapper[hashable_parameters] = EmbeddingsElmo(elmo_path, input_vocabulary, clear_text)
    return _elmo_embeddings_wrapper[hashable_parameters]


_elmo_models_map: Dict[str, Tuple[str, str]] = {
    "small":    ("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
                 "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"),
    "medium":   ("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json",
                 "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"),
    "original": ("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                 "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
}
