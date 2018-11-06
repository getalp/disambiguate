import json
from getalp.wsd.common import get_pretrained_embeddings, get_vocabulary_size
from typing import List
import numpy as np
import os

class ModelConfig(object):

    def __init__(self):
        self.config_root_path: str = str()
        self.input_features: int = int()
        self.input_vocabulary_sizes: List[int] = []
        self.input_embeddings: List[np.array] = []
        self.input_embeddings_sizes: List[int] = []
        self.output_features: int = int()
        self.output_vocabulary_sizes: List[int] = []
        self.lstm_units_size: int = int()
        self.lstm_layers: int = int()
        self.linear_before_lstm: bool = bool()
        self.dropout_rate_before_lstm: float = float()
        self.dropout_rate: float = float()
        self.word_dropout_rate: float = float()
        self.attention_layer: bool = bool()
        self.legacy_model: bool = bool()

    def load_from_file(self, file_path):
        file = open(file_path, "r")
        data = json.load(file)
        file.close()
        self.load_from_serializable_data(data, os.path.dirname(os.path.abspath(file_path)))

    def load_from_serializable_data(self, data, config_root_path):
        self.config_root_path = config_root_path
        self.input_features = data["input_features"]
        self.load_input_vocabularies()
        self.load_input_embeddings(data)
        self.output_features = data["output_features"]
        self.load_output_vocabulary()
        self.lstm_units_size = data["lstm_units_size"]
        self.lstm_layers = data["lstm_layers"]
        self.linear_before_lstm = data["linear_before_lstm"]
        self.dropout_rate_before_lstm = data["dropout_rate_before_lstm"]
        self.dropout_rate = data["dropout_rate"]
        self.word_dropout_rate = data["word_dropout_rate"]
        self.attention_layer = data["attention_layer"]
        self.legacy_model = data["legacy_model"]

    def load_input_vocabularies(self):
        for i in range(0, self.input_features):
            self.input_vocabulary_sizes.append(get_vocabulary_size(self.config_root_path + "/input_vocabulary" + str(i)))

    def load_input_embeddings(self, data):
        input_embeddings_paths = data["input_embeddings_path"]
        if input_embeddings_paths is None:
            input_embeddings_paths = []
        elif isinstance(input_embeddings_paths, str):
            input_embeddings_paths = [input_embeddings_paths]
        self.input_embeddings = []
        for input_embeddings_path in input_embeddings_paths:
            if input_embeddings_path is None:
                self.input_embeddings.append(None)
            elif os.path.isabs(input_embeddings_path):
                self.input_embeddings.append(get_pretrained_embeddings(input_embeddings_path))
            else:
                self.input_embeddings.append(get_pretrained_embeddings(self.config_root_path + "/" + input_embeddings_path))
        for i in range(len(self.input_embeddings), len(self.input_vocabulary_sizes)):
            self.input_embeddings.append(None)

        self.input_embeddings_sizes = data["input_embeddings_size"]
        if self.input_embeddings_sizes is None:
            self.input_embeddings_sizes = []
        elif isinstance(self.input_embeddings_sizes, int):
            self.input_embeddings_sizes = [self.input_embeddings_sizes]
        for i in range(len(self.input_embeddings_sizes), len(self.input_embeddings)):
            self.input_embeddings_sizes.append(None)
        for i in range(0, len(self.input_embeddings_sizes)):
            if self.input_embeddings[i] is not None:
                self.input_embeddings_sizes[i] = self.input_embeddings[i].shape[1]

    def load_output_vocabulary(self):
        for i in range(0, self.output_features):
            self.output_vocabulary_sizes.append(get_vocabulary_size(self.config_root_path + "/output_vocabulary" + str(i)))

