import json
from getalp.common.common import get_value_as_int_list, pad_list, get_value_as_str_list
from getalp.wsd.data_config import DataConfig
from typing import List, Optional


class ModelConfig(object):

    def __init__(self, data_config: DataConfig):
        self.data_config: DataConfig = data_config
        self.input_embeddings_sizes: List[Optional[int]] = []
        self.input_embeddings_tokenize_model: List[str] = []
        self.input_elmo_path: List[str] = []
        self.input_bert_path: List[str] = []
        self.input_auto_model: List[str] = []
        self.input_auto_path: List[str] = []
        self.input_word_dropout_rate: float = float()
        self.input_resize: List[int] = []
        self.input_linear_size: int = int()
        self.input_dropout_rate: float = float()
        self.encoder_type: str = str()
        self.encoder_lstm_hidden_size: int = int()
        self.encoder_lstm_layers: int = int()
        self.encoder_lstm_dropout: float = float()
        self.encoder_transformer_hidden_size: int = int()
        self.encoder_transformer_layers: int = int()
        self.encoder_transformer_heads: int = int()
        self.encoder_transformer_dropout: float = float()
        self.encoder_transformer_positional_encoding: bool = bool()
        self.encoder_transformer_scale_embeddings: bool = bool()
        self.encoder_output_size: int = int()
        self.decoder_translation_transformer_hidden_size = int()
        self.decoder_translation_transformer_layers = int()
        self.decoder_translation_transformer_heads = int()
        self.decoder_translation_transformer_dropout = float()
        self.decoder_translation_scale_embeddings: bool = bool()
        self.decoder_translation_share_embeddings = bool()
        self.decoder_translation_share_encoder_embeddings = bool()
        self.decoder_translation_tokenizer_bert = str()

    def load_from_file(self, file_path):
        file = open(file_path, "r")
        data = json.load(file)
        file.close()
        self.load_from_serializable_data(data)

    def load_from_serializable_data(self, data):
        self.set_input_embeddings_tokenize_model(data.get("input_embeddings_tokenize_model", None))
        self.set_input_elmo_path(data.get("input_elmo_path", None))
        self.set_input_bert_model(data.get("input_bert_path", None))
        self.set_input_auto_model(data.get("input_auto_model", None), data.get("input_auto_path", None))
        # self.set_input_bert_model(None)
        # self.set_input_auto_model(data.get("input_bert_path", data.get("input_auto_path", None)))
        self.load_input_embeddings_sizes(data)
        self.input_word_dropout_rate = data.get("input_word_dropout_rate", None)
        self.set_input_resize(data.get("input_resize", None))
        self.input_linear_size = data.get("input_linear_size", None)
        self.input_dropout_rate = data.get("input_dropout_rate", None)
        self.encoder_type = data.get("encoder_type", "lstm")
        self.encoder_lstm_hidden_size = data.get("encoder_lstm_hidden_size", 1000)
        self.encoder_lstm_layers = data.get("encoder_lstm_layers", 1)
        self.encoder_lstm_dropout = data.get("encoder_lstm_dropout", 0.5)
        self.encoder_transformer_hidden_size = data.get("encoder_transformer_hidden_size", 512)
        self.encoder_transformer_layers = data.get("encoder_transformer_layers", 6)
        self.encoder_transformer_heads = data.get("encoder_transformer_heads", 8)
        self.encoder_transformer_dropout = data.get("encoder_transformer_dropout", 0.1)
        self.encoder_transformer_positional_encoding = data.get("encoder_transformer_positional_encoding", True)
        self.encoder_transformer_scale_embeddings = data.get("encoder_transformer_scale_embeddings", True)
        self.decoder_translation_transformer_hidden_size = data.get("decoder_translation_transformer_hidden_size", 512)
        self.decoder_translation_transformer_layers = data.get("decoder_translation_transformer_layers", 6)
        self.decoder_translation_transformer_heads = data.get("decoder_translation_transformer_heads", 8)
        self.decoder_translation_transformer_dropout = data.get("decoder_translation_transformer_dropout", 0.1)
        self.decoder_translation_scale_embeddings = data.get("decoder_translation_scale_embeddings", True)
        self.decoder_translation_share_embeddings = data.get("decoder_translation_share_embeddings", False)
        self.decoder_translation_share_encoder_embeddings = data.get("decoder_translation_share_encoder_embeddings", False)
        self.decoder_translation_tokenizer_bert = data.get("decoder_translation_tokenizer_bert", None)

    def load_input_embeddings_sizes(self, data):
        self.input_embeddings_sizes = get_value_as_int_list(data.get("input_embeddings_size", None))
        pad_list(self.input_embeddings_sizes, self.data_config.input_features, 300)
        self.reset_input_embeddings_sizes()

    def set_input_embeddings_tokenize_model(self, tokenize_model):
        self.input_embeddings_tokenize_model = get_value_as_str_list(tokenize_model)
        pad_list(self.input_embeddings_tokenize_model, self.data_config.input_features, None)

    def set_input_elmo_path(self, elmo_path):
        self.input_elmo_path = get_value_as_str_list(elmo_path)
        pad_list(self.input_elmo_path, self.data_config.input_features, None)
        self.reset_input_embeddings_sizes()

    def set_input_bert_model(self, bert_model):
        self.input_bert_path = get_value_as_str_list(bert_model)
        pad_list(self.input_bert_path, self.data_config.input_features, None)
        self.reset_input_embeddings_sizes()

    def set_input_auto_model(self, auto_model, auto_path):
        self.input_auto_model = get_value_as_str_list(auto_model)
        self.input_auto_path = get_value_as_str_list(auto_path)
        pad_list(self.input_auto_model, self.data_config.input_features, None)
        pad_list(self.input_auto_path, self.data_config.input_features, None)
        self.reset_input_embeddings_sizes()

    def reset_input_embeddings_sizes(self):
        for i in range(len(self.input_embeddings_sizes)):
            if self.data_config.input_embeddings[i] is not None:
                self.input_embeddings_sizes[i] = self.data_config.input_embeddings[i].shape[1]
            if self.input_elmo_path[i] is not None \
               or self.input_bert_path[i] is not None \
               or self.input_auto_path[i] is not None:
                self.input_embeddings_sizes[i] = None

    def set_input_resize(self, input_resize):
        self.input_resize = get_value_as_int_list(input_resize)
        pad_list(self.input_resize, self.data_config.input_features, None)
        for i in range(len(self.input_resize)):
            if self.input_resize[i] is not None and self.input_resize[i] <= 0:
                self.input_resize[i] = None

    def get_serializable_data(self):
        data = {
            "input_embeddings_size": self.input_embeddings_sizes,
            "input_embeddings_tokenize_model": self.input_embeddings_tokenize_model,
            "input_elmo_path": self.input_elmo_path,
            "input_bert_path": self.input_bert_path,
            "input_auto_model": self.input_auto_model,
            "input_auto_path": self.input_auto_path,
            "input_word_dropout_rate": self.input_word_dropout_rate,
            "input_resize": self.input_resize,
            "input_linear_size": self.input_linear_size,
            "input_dropout_rate": self.input_dropout_rate,
            "encoder_type": self.encoder_type,
            "encoder_lstm_hidden_size": self.encoder_lstm_hidden_size,
            "encoder_lstm_layers": self.encoder_lstm_layers,
            "encoder_lstm_dropout": self.encoder_lstm_dropout,
            "encoder_transformer_hidden_size": self.encoder_transformer_hidden_size,
            "encoder_transformer_layers": self.encoder_transformer_layers,
            "encoder_transformer_heads": self.encoder_transformer_heads,
            "encoder_transformer_dropout": self.encoder_transformer_dropout,
            "encoder_transformer_positional_encoding": self.encoder_transformer_positional_encoding,
            "encoder_transformer_scale_embeddings": self.encoder_transformer_scale_embeddings,
            "decoder_translation_transformer_hidden_size": self.decoder_translation_transformer_hidden_size,
            "decoder_translation_transformer_layers": self.decoder_translation_transformer_layers,
            "decoder_translation_transformer_heads": self.decoder_translation_transformer_heads,
            "decoder_translation_transformer_dropout": self.decoder_translation_transformer_dropout,
            "decoder_translation_scale_embeddings": self.decoder_translation_scale_embeddings,
            "decoder_translation_share_embeddings": self.decoder_translation_share_embeddings,
            "decoder_translation_share_encoder_embeddings": self.decoder_translation_share_encoder_embeddings,
            "decoder_translation_tokenizer_bert": self.decoder_translation_tokenizer_bert
        }
        return data
