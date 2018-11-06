from torch.nn import Module, Embedding, LSTM, Dropout, Linear, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam
import torch
import numpy as np
from getalp.wsd.modules.attention import Attention
from getalp.wsd.model_config import ModelConfig
import random

cpu_device = torch.device("cpu")

if torch.cuda.is_available():
    gpu_device = torch.device("cuda:0")
    default_device = gpu_device
else:
    default_device = cpu_device


class Model(object):

    def __init__(self):
        self.config: ModelConfig = ModelConfig()
        self.backend: TorchModel = None

    def create_model(self):
        self.backend = TorchModel(self.config)

    def set_learning_rate(self, learning_rate):
        self.backend.optimizer = Adam(filter(lambda p: p.requires_grad, self.backend.parameters()), lr=learning_rate)

    def load_model_weights(self, file_path):
        self.backend.load_state_dict(torch.load(file_path, map_location=str(default_device)), strict=True)

    def save_model_weights(self, file_path):
        torch.save(self.backend.state_dict(), file_path)

    def begin_train_on_batch(self):
        self.backend.optimizer.zero_grad()

    def train_on_batch(self, batch_x, batch_y, batch_z):
        self.backend.train()
        self.zero_random_tokens(batch_x, self.config.word_dropout_rate)
        losses, total_loss = self.forward_and_compute_loss(batch_x, batch_y, batch_z)
        total_loss.backward()
        return losses

    def end_train_on_batch(self):
        self.backend.optimizer.step()

    def predict_model_on_batch(self, batch_x):
        self.backend.eval()
        batch_x = self.convert_batch_on_default_device(batch_x)
        output = self.backend(batch_x)
        output = softmax(output[0], dim=2)
        return output.detach().cpu().numpy()

    def predict_model_on_sample(self, sample_x):
        return self.predict_model_on_batch([np.expand_dims(x, axis=0) for x in sample_x])[0]

    def test_model_on_batch(self, batch_x, batch_y, batch_z):
        self.backend.eval()
        losses, total_loss = self.forward_and_compute_loss(batch_x, batch_y, batch_z)
        return losses

    def forward_and_compute_loss(self, batch_x, batch_y, batch_z):
        batch_x = self.convert_batch_on_default_device(batch_x)
        outputs = self.backend(batch_x)
        losses = []
        total_loss = None
        for i in range(len(batch_y)):
            batch_y[i] = torch.from_numpy(batch_y[i]).to(default_device)
            feature_outputs = outputs[i].view(-1, outputs[i].shape[2])
            feature_batch_y = batch_y[i].view(-1)
            loss = self.backend.criterion(feature_outputs, feature_batch_y)
            losses.append(loss.item())
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        return losses, total_loss

    @staticmethod
    def convert_batch_on_default_device(batch):
        for i in range(len(batch)):
            batch[i] = torch.from_numpy(batch[i]).to(default_device)
        return batch

    @staticmethod
    def zero_random_tokens(batch, proba):
        if proba is None: return
        for i in range(len(batch[0])):
            if random.random() < proba:
                for j in range(len(batch)):
                    batch[j][i] = 0


class TorchModel(Module):

    def __init__(self, config):
        super(TorchModel, self).__init__()

        resulting_embeddings_size = 0
        self.embeddings = []
        for i in range(0, config.input_features):
            resulting_embeddings_size += config.input_embeddings_sizes[i]
            if config.input_embeddings[i] is not None:
                module = Embedding.from_pretrained(embeddings=torch.from_numpy(config.input_embeddings[i]).to(default_device), freeze=True)
                if not config.legacy_model:
                    self.add_module("input_embedding" + str(i), module)
            else:
                module = Embedding(num_embeddings=config.input_vocabulary_sizes[i], embedding_dim=config.input_embeddings_sizes[i], padding_idx=0)
                self.add_module("input_embedding" + str(i), module)
            self.embeddings.append(module)

        if config.linear_before_lstm:
            self.linear_before_lstm = Linear(in_features=resulting_embeddings_size, out_features=resulting_embeddings_size)
            self.add_module("linear_before_lstm", self.linear_before_lstm)
        else:
            self.linear_before_lstm = None

        if config.dropout_rate_before_lstm is not None:
            self.dropout_before_lstm = Dropout(p=config.dropout_rate_before_lstm)
            self.add_module("dropout_before_lstm", self.dropout_before_lstm)
        else:
            self.dropout_before_lstm = None

        self.lstm = LSTM(input_size=resulting_embeddings_size, hidden_size=config.lstm_units_size,
                         num_layers=config.lstm_layers, bidirectional=True, batch_first=True)
        self.add_module("lstm", self.lstm)

        if config.dropout_rate is not None:
            self.dropout = Dropout(p=config.dropout_rate)
            self.add_module("dropout", self.dropout)
        else:
            self.dropout = None

        if config.attention_layer is True:
            self.attention = Attention(in_features=config.lstm_units_size * 2)
            self.add_module("attention", self.attention)
            next_layer_in_features = config.lstm_units_size * 4
        else:
            self.attention = None
            next_layer_in_features = config.lstm_units_size * 2

        self.output_linears = []
        if config.legacy_model:
            module = Linear(in_features=next_layer_in_features, out_features=config.output_vocabulary_sizes[0])
            self.output_linears.append(module)
            self.add_module("linear", module)
        else:
            for i in range(0, config.output_features):
                module = Linear(in_features=next_layer_in_features, out_features=config.output_vocabulary_sizes[i])
                self.output_linears.append(module)
                self.add_module("output_linear" + str(i), module)


        if torch.cuda.is_available():
            self.cuda()

        self.criterion = CrossEntropyLoss(ignore_index=0)

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()))


    def forward(self, inputs):
        for i in range(0, len(inputs)):
            inputs[i] = self.embeddings[i](inputs[i])
        inputs = torch.cat(inputs, dim=2)
        if self.linear_before_lstm is not None:
            inputs = self.linear_before_lstm(inputs)
        if self.dropout_before_lstm is not None:
            inputs = self.dropout_before_lstm(inputs)
        inputs, _ = self.lstm(inputs)
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        if self.attention is not None:
            inputs = self.attention(inputs)
        outputs = []
        for i in range(0, len(self.output_linears)):
            outputs.append(self.output_linears[i](inputs))
        return outputs


