import json
from getalp.wsd.common import get_vocabulary, get_vocabulary_size, get_pretrained_embeddings
from getalp.common.common import get_value_as_str_list, get_value_as_bool_list, pad_list
from typing import List
import os
import numpy as np


class DataConfig(object):

    def __init__(self):
        self.config_root_path: str = str()
        self.input_features: int = int()
        self.input_vocabularies: List[List[str]] = []
        self.input_vocabulary_sizes: List[int] = []
        self.input_embeddings_path: List[str] = []
        self.input_embeddings: List[np.array] = []
        self.input_clear_text: List[bool] = []
        self.output_features: int = int()
        self.output_feature_names: List[str] = []
        self.output_vocabulary_sizes: List[int] = []
        self.output_translations: int = 0
        self.output_translation_features: int = 0
        self.output_translation_vocabularies: List[List[List[str]]] = []
        self.output_translation_vocabulary_sizes: List[List[int]] = []
        self.output_translation_clear_text: bool = bool()

    def load_from_file(self, file_path):
        file = open(file_path, "r")
        data = json.load(file)
        file.close()
        self.load_from_serializable_data(data, os.path.dirname(os.path.abspath(file_path)))

    def load_from_serializable_data(self, data, config_root_path):
        self.config_root_path = config_root_path
        self.input_features = data["input_features"]
        self.load_input_vocabularies()
        self.load_input_embeddings_path(data)
        self.load_input_embeddings()
        self.load_input_clear_text_values(data)
        self.output_features = data["output_features"]
        self.output_feature_names = data["output_annotation_name"]
        self.load_output_vocabulary()
        self.output_translations = data.get("output_translations", 0)
        self.output_translation_features = data.get("output_translations", 1)
        self.load_translation_output_vocabulary()
        self.output_translation_clear_text = data.get("output_translation_clear_text", False)

    def load_input_vocabularies(self):
        for i in range(0, self.input_features):
            vocab = get_vocabulary(self.config_root_path + "/input_vocabulary" + str(i))
            self.input_vocabularies.append(vocab)
            self.input_vocabulary_sizes.append(len(vocab))

    def load_input_embeddings_path(self, data):
        self.input_embeddings_path = get_value_as_str_list(data.get("input_embeddings_path", None))
        self.input_embeddings_path = [get_real_path(path, self.config_root_path) for path in self.input_embeddings_path]
        pad_list(self.input_embeddings_path, self.input_features, None)

    def load_input_embeddings(self):
        self.input_embeddings = [None if path is None else get_pretrained_embeddings(path) for path in self.input_embeddings_path]

    def load_input_clear_text_values(self, data):
        self.input_clear_text = get_value_as_bool_list(data.get("input_clear_text", None))
        pad_list(self.input_clear_text, self.input_features, False)

    def load_output_vocabulary(self):
        for i in range(0, self.output_features):
            self.output_vocabulary_sizes.append(get_vocabulary_size(self.config_root_path + "/output_vocabulary" + str(i)))

    def load_translation_output_vocabulary(self):
        for i in range(self.output_translations):
            vocabs: List[List[str]] = []
            vocab_sizes: List[int] = []
            for j in range(self.output_translation_features):
                vocab = get_vocabulary(self.config_root_path + "/output_translation" + str(i) + "_vocabulary" + str(j))
                vocabs.append(vocab)
                vocab_sizes.append(len(vocab))
            self.output_translation_vocabularies.append(vocabs)
            self.output_translation_vocabulary_sizes.append(vocab_sizes)


def get_real_path(path, root_path):
    if path is None:
        return None
    elif os.path.isabs(path):
        return path
    else:
        return root_path + "/" + path
