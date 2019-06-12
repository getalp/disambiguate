from getalp.wsd.common import *
from getalp.wsd.model import Model, ModelConfig, DataConfig
from getalp.common.common import create_directory_if_not_exists, set_if_not_none
import os
import sys
import shutil
import pprint
import sacrebleu
import random
try:
    import tensorboardX
except ImportError:
    tensorboardX = None


class Trainer(object):

    def __init__(self):
        self.data_path: str = str()
        self.model_path: str = str()
        self.batch_size = int()
        self.token_per_batch = int()
        self.eval_frequency = int()
        self.update_every_batch: int = int()
        self.stop_after_epoch = int()
        self.ensemble_size = int()
        self.save_best_loss = bool()
        self.save_end_of_epoch = bool()
        self.shuffle_train_on_init = bool()
        self.warmup_batch_count: int = int()
        self.input_embeddings_size: List[int] = None
        self.input_elmo_model: List[str] = None
        self.input_bert_model: List[str] = None
        self.input_flair_model: List[str] = None
        self.input_word_dropout_rate: float = None
        self.input_resize: List[int] = None
        self.input_apply_linear: bool = None
        self.input_linear_size: int = None
        self.input_dropout_rate: float = None
        self.encoder_type: str = None
        self.encoder_lstm_hidden_size: int = None
        self.encoder_lstm_layers: int = None
        self.encoder_lstm_dropout: float = None
        self.encoder_transformer_hidden_size: int = None
        self.encoder_transformer_layers: int = None
        self.encoder_transformer_heads: int = None
        self.encoder_transformer_dropout: float = None
        self.encoder_transformer_positional_encoding: bool = None
        self.encoder_transformer_scale_embeddings: bool = None
        self.decoder_translation_transformer_hidden_size: int = None
        self.decoder_translation_transformer_dropout: float = None
        self.decoder_translation_scale_embeddings: bool = None
        self.decoder_translation_share_embeddings: bool = None
        self.decoder_translation_share_encoder_embeddings: bool = None
        self.decoder_translation_tokenizer_bert: str = None
        self.optimizer: str = str()
        self.adam_beta1: float = float()
        self.adam_beta2: float = float()
        self.adam_eps: float = float()
        self.lr_scheduler: str = str()
        self.lr_scheduler_fixed_lr: float = float()
        self.lr_scheduler_noam_warmup: int = int()
        self.lr_scheduler_noam_model_size: int = int()
        self.reset: bool = bool()

    def train(self):
        model_weights_last_path = self.model_path + "/model_weights_last"
        model_weights_loss_path = self.model_path + "/model_weights_loss"
        model_weights_wsd_path = self.model_path + "/model_weights_wsd"
        model_weights_bleu_path = self.model_path + "/model_weights_bleu"
        model_weights_end_of_epoch_path = self.model_path + "/model_weights_end_of_epoch_"
        training_info_path = self.model_path + "/training_info"
        tensorboard_path = self.model_path + "/tensorboard"
        train_file_path = self.data_path + "/train"
        dev_file_path = self.data_path + "/dev"
        config_file_path = self.data_path + "/config.json"

        print("Loading config and embeddings")
        data_config: DataConfig = DataConfig()
        data_config.load_from_file(config_file_path)
        config: ModelConfig = ModelConfig(data_config)
        config.load_from_file(config_file_path)

        # change config from CLI parameters
        config.input_embeddings_sizes = set_if_not_none(self.input_embeddings_size, config.input_embeddings_sizes)
        if self.input_elmo_model is not None:
            config.set_input_elmo_path(self.input_elmo_model)
        if self.input_bert_model is not None:
            config.set_input_bert_model(self.input_bert_model)
        if self.input_flair_model is not None:
            config.set_input_flair_model(self.input_flair_model)
        if self.input_word_dropout_rate is not None:
            config.input_word_dropout_rate = self.input_word_dropout_rate
            eprint("Warning: input_word_dropout_rate is not implemented")
        config.input_apply_linear = set_if_not_none(self.input_apply_linear, config.input_apply_linear)
        config.input_linear_size = set_if_not_none(self.input_linear_size, config.input_linear_size)
        config.input_dropout_rate = set_if_not_none(self.input_dropout_rate, config.input_dropout_rate)
        config.encoder_type = set_if_not_none(self.encoder_type, config.encoder_type)
        config.encoder_lstm_hidden_size = set_if_not_none(self.encoder_lstm_hidden_size, config.encoder_lstm_hidden_size)
        config.encoder_lstm_layers = set_if_not_none(self.encoder_lstm_layers, config.encoder_lstm_layers)
        config.encoder_lstm_dropout = set_if_not_none(self.encoder_lstm_dropout, config.encoder_lstm_dropout)
        config.encoder_transformer_hidden_size = set_if_not_none(self.encoder_transformer_hidden_size, config.encoder_transformer_hidden_size)
        config.encoder_transformer_layers = set_if_not_none(self.encoder_transformer_layers, config.encoder_transformer_layers)
        config.encoder_transformer_heads = set_if_not_none(self.encoder_transformer_heads, config.encoder_transformer_heads)
        config.encoder_transformer_dropout = set_if_not_none(self.encoder_transformer_dropout, config.encoder_transformer_dropout)
        config.encoder_transformer_positional_encoding = set_if_not_none(self.encoder_transformer_positional_encoding, config.encoder_transformer_positional_encoding)
        config.encoder_transformer_scale_embeddings = set_if_not_none(self.encoder_transformer_scale_embeddings, config.encoder_transformer_scale_embeddings)
        config.decoder_translation_transformer_hidden_size = set_if_not_none(self.decoder_translation_transformer_hidden_size, config.decoder_translation_transformer_hidden_size)
        config.decoder_translation_transformer_dropout = set_if_not_none(self.decoder_translation_transformer_dropout, config.decoder_translation_transformer_dropout)
        config.decoder_translation_scale_embeddings = set_if_not_none(self.decoder_translation_scale_embeddings, config.decoder_translation_scale_embeddings)
        config.decoder_translation_share_embeddings = set_if_not_none(self.decoder_translation_share_embeddings, config.decoder_translation_share_embeddings)
        config.decoder_translation_share_encoder_embeddings = set_if_not_none(self.decoder_translation_share_encoder_embeddings, config.decoder_translation_share_encoder_embeddings)
        config.decoder_translation_tokenizer_bert = set_if_not_none(self.decoder_translation_tokenizer_bert, config.decoder_translation_tokenizer_bert)

        print("GPU is available: " + str(torch.cuda.is_available()))

        model: Model = Model(config)
        model.set_adam_parameters(adam_beta1=self.adam_beta1, adam_beta2=self.adam_beta2, adam_eps=self.adam_eps)
        model.set_lr_scheduler(lr_scheduler=self.lr_scheduler, fixed_lr=self.lr_scheduler_fixed_lr, warmup=self.lr_scheduler_noam_warmup, model_size=self.lr_scheduler_noam_model_size)

        current_ensemble = 0
        current_epoch = 0
        current_batch = 0
        current_batch_total = 0
        current_sample_index = 0
        best_dev_loss = None
        best_dev_wsd = None
        best_dev_bleu = None
        random_seed = self.generate_random_seed()

        if not self.reset and os.path.isfile(training_info_path) and os.path.isfile(model_weights_last_path):
            print("Resuming from previous training")
            current_ensemble, current_epoch, current_batch, current_batch_total, current_sample_index, best_dev_loss, best_dev_wsd, best_dev_bleu, random_seed = load_training_info(training_info_path)
            model.load_model_weights(model_weights_last_path)
        else:
            print("Creating model")
            model.create_model()
            create_directory_if_not_exists(self.model_path)

        print("Random seed is " + str(random_seed))

        print("Config is: ")
        pprint.pprint(config.get_serializable_data())

        print("Number of parameters (total): " + model.get_number_of_parameters(filter_requires_grad=False))
        print("Number of parameters (learned): " + model.get_number_of_parameters(filter_requires_grad=True))

        print("Warming up on " + str(self.warmup_batch_count) + " batches")
        train_samples = read_samples_from_file(train_file_path, data_config.input_clear_text, data_config.output_features, data_config.output_translations, data_config.output_translation_features, data_config.output_translation_clear_text, self.batch_size*self.warmup_batch_count)
        model.preprocess_samples(train_samples)
        for i in range(self.warmup_batch_count):
            batch_x, batch_y, batch_z, batch_tt, actual_batch_size, reached_eof = read_batch_from_samples(train_samples, self.batch_size, -1, 0, data_config.input_features, data_config.output_features, data_config.output_translations, data_config.output_translation_features, data_config.input_clear_text, data_config.output_translation_clear_text)
            model.begin_train_on_batch()
            model.train_on_batch(batch_x, batch_y, batch_tt)
            model.end_train_on_batch()

        print("Loading training and development data")
        train_samples = read_samples_from_file(train_file_path, input_clear_text=data_config.input_clear_text, output_features=data_config.output_features, output_translations=data_config.output_translations, output_translation_features=data_config.output_translation_features, output_translation_clear_text=data_config.output_translation_clear_text)
        dev_samples = read_samples_from_file(dev_file_path, input_clear_text=data_config.input_clear_text, output_features=data_config.output_features, output_translations=data_config.output_translations, output_translation_features=data_config.output_translation_features, output_translation_clear_text=data_config.output_translation_clear_text)

        print("Preprocessing training and development data")
        model.preprocess_samples(train_samples)
        model.preprocess_samples(dev_samples)

        if self.shuffle_train_on_init:
            print("Shuffling training data")
            random.seed(random_seed)
            random.shuffle(train_samples)

        self.print_state(current_ensemble, current_epoch, current_batch, current_batch_total, len(train_samples), current_sample_index, [None for _ in range(data_config.output_features + data_config.output_translations * data_config.output_translation_features)], [None for _ in range(data_config.output_features + data_config.output_translations * data_config.output_translation_features)], [None for _ in range(data_config.output_features)], None)

        if self.reset:
            shutil.rmtree(tensorboard_path, ignore_errors=True)

        for current_ensemble in range(current_ensemble, self.ensemble_size):
            if tensorboardX is not None:
                tb_writer = tensorboardX.SummaryWriter(tensorboard_path + '/ensemble' + str(current_ensemble))
            else:
                tb_writer = None
            sample_accumulate_between_eval = 0
            train_losses = None
            while self.stop_after_epoch == -1 or current_epoch < self.stop_after_epoch:

                model.update_learning_rate(step=current_batch_total)

                print("training sample " + str(current_sample_index) + "/" + str(len(train_samples)), end="\r")
                sys.stdout.flush()

                reached_eof = False
                model.begin_train_on_batch()
                for _ in range(self.update_every_batch):
                    batch_x, batch_y, batch_z, batch_tt, actual_batch_size, reached_eof = read_batch_from_samples(train_samples, self.batch_size, self.token_per_batch, current_sample_index, data_config.input_features, data_config.output_features, data_config.output_translations, data_config.output_translation_features, data_config.input_clear_text, data_config.output_translation_clear_text)
                    if actual_batch_size == 0:
                        break
                    batch_losses = model.train_on_batch(batch_x, batch_y, batch_tt)
                    if train_losses is None:
                        train_losses = [0 for _ in batch_losses]
                    for i in range(len(batch_losses)):
                        train_losses[i] += batch_losses[i] * actual_batch_size
                    current_sample_index += actual_batch_size
                    sample_accumulate_between_eval += actual_batch_size
                    current_batch += 1
                    current_batch_total += 1
                    if reached_eof:
                        break
                model.end_train_on_batch()

                if reached_eof:
                    if self.save_end_of_epoch:
                        model.save_model_weights(model_weights_end_of_epoch_path + str(current_epoch) + "_" + str(current_ensemble))
                    current_batch = 0
                    current_sample_index = 0
                    current_epoch += 1
                    random_seed = self.generate_random_seed()
                    random.seed(random_seed)
                    random.shuffle(train_samples)

                if current_batch % self.eval_frequency == 0:
                    dev_losses, dev_wsd, dev_bleu = self.test_on_dev(dev_samples, model)
                    for i in range(len(train_losses)):
                        train_losses[i] /= float(sample_accumulate_between_eval)
                    self.print_state(current_ensemble, current_epoch, current_batch, current_batch_total, len(train_samples), current_sample_index, train_losses, dev_losses, dev_wsd, dev_bleu)
                    self.write_tensorboard(tb_writer, current_epoch, train_samples, current_sample_index, train_losses, dev_losses, dev_wsd, data_config.output_feature_names, dev_bleu, model.optimizer.scheduler.get_learning_rate(current_batch_total))
                    sample_accumulate_between_eval = 0
                    train_losses = None

                    if best_dev_loss is None or dev_losses[0] < best_dev_loss:
                        if self.save_best_loss:
                            model.save_model_weights(model_weights_loss_path + str(current_ensemble))
                            print("New best dev loss: " + str(dev_losses[0]))
                        best_dev_loss = dev_losses[0]

                    if len(dev_wsd) > 0 and (best_dev_wsd is None or dev_wsd[0] > best_dev_wsd):
                        model.save_model_weights(model_weights_wsd_path + str(current_ensemble))
                        best_dev_wsd = dev_wsd[0]
                        print("New best dev WSD: " + str(best_dev_wsd))

                    if (best_dev_bleu is None or dev_bleu > best_dev_bleu) and dev_bleu is not None:
                        model.save_model_weights(model_weights_bleu_path + str(current_ensemble))
                        best_dev_bleu = dev_bleu
                        print("New best dev BLEU: " + str(best_dev_bleu))

                    model.save_model_weights(model_weights_last_path)
                    save_training_info(training_info_path, current_ensemble, current_epoch, current_batch, current_batch_total, current_sample_index, best_dev_loss, best_dev_wsd, best_dev_bleu, random_seed)

            model.create_model()
            current_epoch = 0
            current_batch_total = 0
            best_dev_loss = None
            best_dev_wsd = None
            best_dev_bleu = None

    @staticmethod
    def generate_random_seed():
        return int.from_bytes(os.urandom(8), byteorder='big', signed=False)

    @staticmethod
    def print_state(current_ensemble, current_epoch, current_batch, current_batch_total, samples_count, current_sample, train_losses, dev_losses, dev_wsd, dev_bleu):
        print("Ensemble " + str(current_ensemble) + " - Epoch " + str(current_epoch) + " - Batch " + str(current_batch) +
              " - Sample " + str(current_sample) + " - Total Batch " + str(current_batch_total) + " - Total Sample " + str(current_epoch * samples_count + current_sample) +
              " - Train losses = " + str(train_losses) +
              " - Dev losses = " + str(dev_losses) + " - Dev wsd = " + str(dev_wsd) + " - Dev bleu = " + str(dev_bleu))

    @staticmethod
    def write_tensorboard(tb_writer, current_epoch, train_samples, current_sample_index, train_losses, dev_losses, dev_wsd, output_annotation_names, dev_bleu, learning_rate):
        if tb_writer is None:
            return
        tb_index = current_epoch * len(train_samples) + current_sample_index
        if len(train_losses) > 0:
            tb_writer.add_scalar('train_loss', train_losses[0], tb_index)
        if len(dev_losses) > 0:
            tb_writer.add_scalar('dev_loss', dev_losses[0], tb_index)
        if len(dev_wsd) > 0:
            tb_writer.add_scalar('dev_wsd', dev_wsd[0], tb_index)
        if dev_bleu is not None:
            tb_writer.add_scalar('dev_bleu', dev_bleu, tb_index)

        tb_writer.add_scalar('learning_rate', learning_rate, tb_index)

        for i in range(len(train_losses)):
            tb_writer.add_scalar('train_losses/train_loss' + str(i), train_losses[i], tb_index)
        for i in range(len(dev_losses)):
            tb_writer.add_scalar('dev_losses/dev_loss' + str(i), dev_losses[i], tb_index)
        for i in range(1, len(dev_wsd)):
            tb_writer.add_scalar('dev_wsds/dev_wsd' + str(i) + "_" + output_annotation_names[i-1].replace("%", "_"), dev_wsd[i], tb_index)

    def test_on_dev(self, dev_samples, model: Model):
        loss = self.get_loss_metrics(dev_samples, model)
        wsd = self.get_wsd_metrics(dev_samples, model)
        bleu = None
        if model.config.data_config.output_translations > 0:
            bleu = self.get_bleu_metrics(dev_samples, model)
        return loss, wsd, bleu

    def get_loss_metrics(self, dev_samples, model: Model):
        losses = [0 for _ in range(model.config.data_config.output_features + model.config.data_config.output_translations * model.config.data_config.output_translation_features)]
        reached_eof = False
        current_index = 0
        while not reached_eof:
            batch_x, batch_y, batch_z, batch_tt, actual_batch_size, reached_eof = read_batch_from_samples(dev_samples, self.batch_size, self.token_per_batch, current_index, model.config.data_config.input_features, model.config.data_config.output_features, model.config.data_config.output_translations, model.config.data_config.output_translation_features, model.config.data_config.input_clear_text, model.config.data_config.output_translation_clear_text)
            if actual_batch_size == 0:
                break
            batch_losses = model.test_model_on_batch(batch_x, batch_y, batch_tt)
            for i in range(len(batch_losses)):
                losses[i] += (batch_losses[i] * actual_batch_size)
            current_index += actual_batch_size
        if current_index != 0:
            for i in range(len(losses)):
                losses[i] /= float(current_index)
        return losses

    def get_wsd_metrics(self, dev_samples, model: Model):
        output_features = model.config.data_config.output_features
        if output_features == 0:
            return []
        goods = [0 for _ in range(output_features)]
        totals = [0 for _ in range(output_features)]
        reached_eof = False
        current_index = 0
        while not reached_eof:
            batch_x, batch_y, batch_z, batch_tt, actual_batch_size, reached_eof = read_batch_from_samples(dev_samples, self.batch_size, self.token_per_batch, current_index, model.config.data_config.input_features, model.config.data_config.output_features, model.config.data_config.output_translations, model.config.data_config.output_translation_features, model.config.data_config.input_clear_text, model.config.data_config.output_translation_clear_text)
            if actual_batch_size == 0:
                break
            output = model.predict_all_features_on_batch(batch_x)
            for k in range(len(output)):  # k: feat
                for i in range(len(output[k])):  # i: batch
                    for j in range(len(output[k][i])):  # j: seq
                        if j < len(batch_z[k][i]):
                            restricted_possibilities = batch_z[k][i][j]
                            max_possibility = None
                            if 0 in restricted_possibilities:
                                max_possibility = None
                            elif -1 in restricted_possibilities:
                                max_possibility = torch_argmax(output[k][i][j]).item()
                            else:
                                max_proba = None
                                for possibility in restricted_possibilities:
                                    proba = output[k][i][j][possibility].item()
                                    if max_proba is None or proba > max_proba:
                                        max_proba = proba
                                        max_possibility = possibility
                            if max_possibility is not None:
                                totals[k] += 1
                                if max_possibility == batch_y[k][i][j].item():
                                    goods[k] += 1
            current_index += actual_batch_size
        all_wsd = [((float(goods[i]) / float(totals[i])) * float(100)) if totals[i] != 0 else float(0) for i in range(output_features)]
        if output_features > 1:
            if sum(totals) != 0:
                summary_wsd = [(float(sum(goods)) / float(sum(totals))) * float(100)]
            else:
                summary_wsd = [float(0)]
            return summary_wsd + all_wsd
        else:
            return all_wsd

    def get_bleu_metrics(self, dev_samples, model: Model):
        reached_eof = False
        current_index = 0
        all_hypothesis_sentences = []
        all_reference_sentences = []
        while not reached_eof:
            batch_x, batch_y, batch_z, batch_tt, actual_batch_size, reached_eof = read_batch_from_samples(dev_samples, self.batch_size, self.token_per_batch, current_index, model.config.data_config.input_features, model.config.data_config.output_features, model.config.data_config.output_translations, model.config.data_config.output_translation_features, model.config.data_config.input_clear_text, model.config.data_config.output_translation_clear_text)
            if actual_batch_size == 0:
                break
            reference = unpad_turn_to_text_and_remove_bpe_of_batch_t(batch_tt[0][0], model.config.data_config.output_translation_vocabularies[0][0])
            for sentence in reference:
                all_reference_sentences.append(sentence)
            output = model.predict_translation_on_batch(batch_x)
            output = unpad_turn_to_text_and_remove_bpe_of_batch_t(output, model.config.data_config.output_translation_vocabularies[0][0])
            for sentence in output:
                all_hypothesis_sentences.append(sentence)
            current_index += actual_batch_size
            if reached_eof is True:
                break
        bleu = sacrebleu.raw_corpus_bleu(sys_stream=all_hypothesis_sentences, ref_streams=[all_reference_sentences])
        return bleu.score
