from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
import torch
from getalp.wsd.loss.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from getalp.wsd.modules.embeddings import get_elmo_embeddings, get_bert_embeddings, get_auto_embeddings, EmbeddingsLUT
from getalp.wsd.modules.encoders import EncoderBase, EncoderLSTM, EncoderTransformer
from getalp.wsd.modules.decoders import DecoderClassify, DecoderTranslateTransformer
from getalp.wsd.optim import SchedulerFixed, SchedulerNoam
from getalp.wsd.model_config import ModelConfig
from getalp.wsd.data_config import DataConfig
from getalp.wsd.torch_utils import default_device
import random
from typing import List, Union, Optional


class Model(object):

    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.backend: Optional[TorchModel] = None
        self.optimizer: TorchModelOptimizer = TorchModelOptimizer()
        self.classification_criterion = CrossEntropyLoss(ignore_index=0)
        self.translation_criterion = LabelSmoothedCrossEntropyCriterion(label_smoothing=0.1, ignore_index=0)

    def create_model(self):
        self.backend = TorchModel(self.config, self.config.data_config)
        self.optimizer.set_backend(self.backend)

    def get_number_of_parameters(self, filter_requires_grad: bool):
        raw_count = sum(p.numel() for p in self.backend.parameters() if not filter_requires_grad or p.requires_grad)
        if raw_count > 1000000:
            str_count = "%.2f" % (float(raw_count) / float(1000000)) + "M"
        elif raw_count > 1000:
            str_count = "%.2f" % (float(raw_count) / float(1000)) + "K"
        else:
            str_count = str(raw_count)
        return str_count

    def set_adam_parameters(self, adam_beta1: float, adam_beta2: float, adam_eps: float):
        self.optimizer.set_adam_parameters(adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps)

    def set_lr_scheduler(self, lr_scheduler: str, fixed_lr: float, warmup: int, model_size: int):
        self.optimizer.set_scheduler(scheduler=lr_scheduler, fixed_lr=fixed_lr, warmup=warmup, model_size=model_size)

    def update_learning_rate(self, step):
        self.optimizer.update_learning_rate(step)

    def set_beam_size(self, beam_size: int):
        if self.backend.decoder_translation is not None:
            self.backend.decoder_translation.beam_size = beam_size

    def load_model_weights(self, file_path):
        save = torch.load(file_path, map_location=str(default_device))
        self.config.load_from_serializable_data(save["config"])
        self.create_model()
        self.backend.encoder.load_state_dict(save["backend_encoder"], strict=True)
        if self.backend.decoder_classification is not None:
            self.backend.decoder_classification.load_state_dict(save["backend_decoder_classification"], strict=True)
        if self.backend.decoder_translation is not None:
            self.backend.decoder_translation.load_state_dict(save["backend_decoder_translation"], strict=True)
        self.optimizer.adam.load_state_dict(save["optimizer"])
        for i in range(len(self.backend.embeddings)):
            if not self.backend.embeddings[i].is_fixed():
                self.backend.embeddings[i].load_state_dict(save["backend_embeddings" + str(i)], strict=True)

    def save_model_weights(self, file_path):
        save = {"config":    self.config.get_serializable_data(),
                "backend_encoder":   self.backend.encoder.state_dict(),
                "optimizer": self.optimizer.adam.state_dict()}
        if self.backend.decoder_classification is not None:
            save["backend_decoder_classification"] = self.backend.decoder_classification.state_dict()
        if self.backend.decoder_translation is not None:
            save["backend_decoder_translation"] = self.backend.decoder_translation.state_dict()
        for i in range(len(self.backend.embeddings)):
            if not self.backend.embeddings[i].is_fixed():
                save["backend_embeddings" + str(i)] = self.backend.embeddings[i].state_dict()
        torch.save(save, file_path)

    def begin_train_on_batch(self):
        self.optimizer.adam.zero_grad()

    def train_on_batch(self, batch_x, batch_y, batch_tt):
        self.backend.train()
        for i in range(len(self.backend.embeddings)):
            if self.backend.embeddings[i].is_fixed():
                self.backend.embeddings[i].eval()
        losses, total_loss = self.forward_and_compute_loss(batch_x, batch_y, batch_tt)
        total_loss.backward()
        return losses

    def end_train_on_batch(self):
        self.optimizer.adam.step()

    @torch.no_grad()
    def predict_wsd_on_batch(self, batch_x):
        self.backend.eval()
        batch_x = self.convert_batch_x_on_default_device(batch_x)
        outputs_classification, _ = self.backend(batch_x, [[None]])
        outputs = outputs_classification[0]
        return outputs

    @torch.no_grad()
    def predict_all_features_on_batch(self, batch_x):
        self.backend.eval()
        batch_x = self.convert_batch_x_on_default_device(batch_x)
        outputs_classification, _ = self.backend(batch_x, [[None]])
        outputs = outputs_classification
        return outputs

    @torch.no_grad()
    def predict_translation_on_batch(self, batch_x):
        self.backend.eval()
        batch_x = self.convert_batch_x_on_default_device(batch_x)
        _, outputs_translation = self.backend(batch_x, [[None]])
        outputs = outputs_translation[0][0]
        return outputs

    @torch.no_grad()
    def predict_wsd_and_translation_on_batch(self, batch_x):
        self.backend.eval()
        batch_x = self.convert_batch_x_on_default_device(batch_x)
        outputs_classification, outputs_translation = self.backend(batch_x, [[None]])
        outputs_classification = outputs_classification[0]
        outputs_translation = outputs_translation[0][0]
        return outputs_classification, outputs_translation

    @torch.no_grad()
    def test_model_on_batch(self, batch_x, batch_y, batch_tt):
        self.backend.eval()
        losses, total_loss = self.forward_and_compute_loss(batch_x, batch_y, batch_tt)
        return losses

    def forward_and_compute_loss(self, batch_x, batch_y, batch_tt):
        batch_x = self.convert_batch_x_on_default_device(batch_x)
        batch_y = self.convert_batch_y_on_default_device(batch_y)
        batch_tt = self.convert_batch_tt_on_default_device(batch_tt)
        outputs_classification, outputs_translation = self.backend(batch_x, batch_tt)
        losses = []
        total_loss = None
        for i in range(len(batch_y)):
            feature_outputs = outputs_classification[i].view(-1, outputs_classification[i].shape[2])
            feature_batch_y = batch_y[i].view(-1)
            loss = self.classification_criterion(feature_outputs, feature_batch_y)
            losses.append(loss.item())
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        if len(batch_tt) > 0:
            outputs_translation[0][0] = outputs_translation[0][0].contiguous()
            translation_output = outputs_translation[0][0].view(-1, outputs_translation[0][0].shape[2])
            translation_batch_tt = batch_tt[0][0].view(-1)
            loss = self.translation_criterion(translation_output, translation_batch_tt)
            losses.append(loss.item())
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        return losses, total_loss

    def convert_batch_x_on_default_device(self, batch_x):
        return [x.to(default_device) if not self.config.data_config.input_clear_text[i] else x for i, x in enumerate(batch_x)]

    @staticmethod
    def convert_batch_y_on_default_device(batch_y):
        return [x.to(default_device) for x in batch_y]

    @staticmethod
    def convert_batch_tt_on_default_device(batch_tt):
        return [[x.to(default_device) for x in y] for y in batch_tt]

    @staticmethod
    def zero_random_tokens(batch, proba):
        if proba is None:
            return
        for i in range(len(batch[0])):
            if random.random() < proba:
                for j in range(len(batch)):
                    batch[j][i] = 0

    # samples : sample x xyztt x feat x batch x seq
    def preprocess_samples(self, samples):
        for sample in samples:
            sample[0][0], new_size, indices = self.backend.embeddings[0].preprocess_sample_first(sample[0][0])
            for i in range(self.config.data_config.input_features):
                sample[0][i] = self.backend.embeddings[i].preprocess_sample_next(sample[0][i], new_size, indices)


class TorchModelOptimizer(object):

    def __init__(self):
        super().__init__()
        self.adam_beta1: float = None
        self.adam_beta2: float = None
        self.adam_eps: float = None
        self.adam: Adam = None
        self.scheduler: Union[SchedulerFixed, SchedulerNoam] = None

    def set_adam_parameters(self, adam_beta1: float, adam_beta2: float, adam_eps: float):
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

    def set_scheduler(self, scheduler: str, fixed_lr: float, warmup: int, model_size: int):
        if scheduler == "noam":
            self.scheduler = SchedulerNoam(warmup=warmup, model_size=model_size)
        else:  # if scheduler == "fixed":
            self.scheduler = SchedulerFixed(fixed_lr=fixed_lr)

    def set_backend(self, backend: Module):
        if self.adam_beta1 is None or self.adam_beta2 is None or self.adam_eps is None:
            self.adam = Adam(filter(lambda p: p.requires_grad, backend.parameters()))
        else:
            self.adam = Adam(filter(lambda p: p.requires_grad, backend.parameters()), betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)

    def update_learning_rate(self, step: int):
        self.set_learning_rate(self.scheduler.get_learning_rate(step))

    def set_learning_rate(self, learning_rate: float):
        for param_group in self.adam.param_groups:
            param_group['lr'] = learning_rate


class TorchModel(Module):

    def __init__(self, config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.config = config

        self.embeddings: List[Module] = []
        for i in range(0, data_config.input_features):
            if config.input_elmo_path[i] is not None:
                module = get_elmo_embeddings(elmo_path=config.input_elmo_path[i], input_vocabulary=data_config.input_vocabularies[i], clear_text=data_config.input_clear_text[i])
            elif config.input_bert_path[i] is not None:
                module = get_bert_embeddings(bert_path=config.input_bert_path[i], clear_text=data_config.input_clear_text[i])
            elif config.input_auto_path[i] is not None:
                module = get_auto_embeddings(auto_model=config.input_auto_model[i], auto_path=config.input_auto_path[i], clear_text=data_config.input_clear_text[i])
            else:
                module = EmbeddingsLUT(input_embeddings=data_config.input_embeddings[i], input_vocabulary_size=data_config.input_vocabulary_sizes[i], input_embeddings_size=config.input_embeddings_sizes[i], clear_text=data_config.input_clear_text[i], tokenize_model=config.input_embeddings_tokenize_model[i])
            config.input_embeddings_sizes[i] = module.get_output_dim()
            self.add_module("input_embedding" + str(i), module)
            self.embeddings.append(module)

        self.encoder: Optional[Module] = None
        if config.encoder_type == "lstm":
            self.encoder = EncoderLSTM(config)
        elif config.encoder_type == "transformer":
            self.encoder = EncoderTransformer(config)
        else:
            self.encoder = EncoderBase(config)

        if data_config.output_features > 0:
            self.decoder_classification = DecoderClassify(config, data_config)
        else:
            self.decoder_classification = None

        if data_config.output_translations > 0:
            self.decoder_translation = DecoderTranslateTransformer(config, data_config, self.embeddings[0])
        else:
            self.decoder_translation = None

        if torch.cuda.is_available():
            self.cuda()

    # inputs:
    #   - List[Union[LongTensor, List[str]]]       features x batch x seq_in                             (input features)
    #   - List[List[LongTensor]]                   translations x features x batch x seq_out             (output translations) (training only)
    # outputs:
    #   - List[FloatTensor]                        features x batch x seq_in x vocab_out                 (output features)
    #   - List[List[FloatTensor]]                  translations x features x batch x seq_out x vocab_out (output translations)
    def forward(self, inputs, translation_true_output):
        inputs[0], pad_mask, token_indices = self.embeddings[0](inputs[0])
        for i in range(1, len(inputs)):
            inputs[i], _, _ = self.embeddings[i](inputs[i])
        inputs = self.encoder(inputs, pad_mask)
        classification_outputs = []
        if self.decoder_classification is not None:
            classification_outputs = self.decoder_classification(inputs, token_indices)
        translation_outputs = []
        if self.decoder_translation is not None:
            translation_outputs = self.decoder_translation(inputs, pad_mask, translation_true_output[0][0])
        return classification_outputs, [[translation_outputs]]
