from getalp.wsd.common import *
from getalp.wsd.model import Model, ModelConfig
from getalp.common.common import create_directory_if_not_exists
import os


class Trainer(object):

    def __init__(self):
        self.data_path: str = str()
        self.model_path: str = str()
        self.batch_size = int()
        self.test_every_batch = int()
        self.stop_after_epoch = int()
        self.ensemble_size = int()
        self.save_best_loss = bool()
        self.save_end_of_epoch = bool()
        self.shuffle_train_on_init = bool()
        self.warmup_sample_size: int = int()
        self.reset: bool = bool()
        self.learning_rate: float = float()
        self.update_every_batch: int = int()


    def train(self):
        model_weights_last_path = self.model_path + "/model_weights_last"
        model_weights_loss_path = self.model_path + "/model_weights_loss"
        model_weights_wsd_path = self.model_path + "/model_weights_wsd"
        model_weights_end_of_epoch_path = self.model_path + "/model_weights_end_of_epoch_"
        training_info_path = self.model_path + "/training_info"
        training_losses_path = self.model_path + "/training_losses"
        train_file_path = self.data_path + "/train"
        dev_file_path = self.data_path + "/dev"
        config_file_path = self.data_path + "/config.json"

        print("Loading config and embeddings")
        config = ModelConfig()
        config.load_from_file(config_file_path)

        print("Creating model")
        model = Model()
        model.config = config
        self.recreate_model(model)

        print("Warming up on fake batch")
        batch_x, batch_y = create_fake_batch(batch_size=self.batch_size, sample_size=self.warmup_sample_size, input_features=model.config.input_features, input_vocabulary_sizes=model.config.input_vocabulary_sizes, output_features=model.config.output_features, output_vocabulary_sizes=model.config.output_vocabulary_sizes)
        model.begin_train_on_batch()
        model.train_on_batch(batch_x, batch_y, None)
        model.end_train_on_batch()

        self.recreate_model(model)

        print("Loading training and development data")
        train_samples = read_all_samples_from_file(train_file_path)
        dev_samples = read_all_samples_from_file(dev_file_path)

        current_ensemble = 0
        current_epoch = 0
        current_batch = 0
        current_sample_index = 0
        best_dev_wsd = None
        best_dev_loss = None

        if not self.reset and os.path.isfile(training_info_path) and os.path.isfile(model_weights_last_path):
            print("Resuming from previous training")
            current_ensemble, current_epoch, current_batch, current_sample_index, best_dev_wsd, best_dev_loss = load_training_info(training_info_path)
            model.load_model_weights(model_weights_last_path)
        elif self.shuffle_train_on_init:
            print("Shuffling training data")
            random.shuffle(train_samples)

        create_directory_if_not_exists(self.model_path)

        self.print_state(current_ensemble, current_epoch, current_batch, [None for _ in range(model.config.output_features)], [None for _ in range(model.config.output_features)], None)

        for current_ensemble in range(current_ensemble, self.ensemble_size):
            sample_accumulate_between_eval = 0
            train_losses = None
            while self.stop_after_epoch == -1 or current_epoch < self.stop_after_epoch:

                reached_eof = False
                model.begin_train_on_batch()
                for _ in range(self.update_every_batch):
                    batch_x, batch_y, batch_z, actual_batch_size, reached_eof = read_batch_from_samples(train_samples, self.batch_size, current_sample_index)
                    if actual_batch_size == 0: break
                    batch_losses = model.train_on_batch(batch_x, batch_y, batch_z)
                    if train_losses is None:
                        train_losses = [0 for _ in batch_losses]
                    for i in range(len(batch_losses)):
                        train_losses[i] += batch_losses[i] * actual_batch_size
                    current_sample_index += actual_batch_size
                    sample_accumulate_between_eval += actual_batch_size
                    current_batch += 1
                    if reached_eof: break
                model.end_train_on_batch()

                if reached_eof:
                    print("Reached eof at batch " + str(current_batch))
                    if self.save_end_of_epoch:
                        model.save_model_weights(model_weights_end_of_epoch_path + str(current_epoch) + "_" + str(current_ensemble))
                    current_batch = 0
                    current_sample_index = 0
                    current_epoch += 1
                    random.shuffle(train_samples)

                if current_batch % self.test_every_batch == 0:
                    dev_wsd, dev_losses = self.test_on_dev(self.batch_size, dev_samples, model)
                    for i in range(len(train_losses)):
                        train_losses[i] /= float(sample_accumulate_between_eval)
                    self.print_state(current_ensemble, current_epoch, current_batch, train_losses, dev_losses, dev_wsd)
                    save_training_losses(training_losses_path, train_losses[0], dev_losses[0], dev_wsd)
                    sample_accumulate_between_eval = 0
                    train_losses = None

                    if best_dev_loss is None or dev_losses[0] < best_dev_loss:
                        if self.save_best_loss:
                            model.save_model_weights(model_weights_loss_path + str(current_ensemble))
                        best_dev_loss = dev_losses[0]

                    if best_dev_wsd is None or dev_wsd > best_dev_wsd:
                        model.save_model_weights(model_weights_wsd_path + str(current_ensemble))
                        best_dev_wsd = dev_wsd
                        print("New best dev WSD: " + str(best_dev_wsd))

                    model.save_model_weights(model_weights_last_path)
                    save_training_info(training_info_path, current_ensemble, current_epoch, current_batch, current_sample_index, best_dev_wsd, best_dev_loss)

            self.recreate_model(model)
            current_epoch = 0
            best_dev_wsd = None
            best_dev_loss = None


    def recreate_model(self, model):
        model.create_model()
        model.set_learning_rate(self.learning_rate)


    @staticmethod
    def print_state(current_ensemble, current_epoch, current_batch, train_losses, dev_losses, dev_wsd):
        print("Ensemble " + str(current_ensemble) + " - Epoch " + str(current_epoch) + " - Batch " + str(current_batch) + " - Train losses = " + str(train_losses) + " - Dev losses = " + str(dev_losses) + " - Dev wsd = " + str(dev_wsd))


    def test_on_dev(self, batch_size, dev_samples, model):
        loss = self.get_loss_metrics(batch_size, dev_samples, model)
        wsd = self.get_wsd_metrics(batch_size, dev_samples, model)
        return wsd, loss


    @staticmethod
    def get_loss_metrics(batch_size, dev_samples, model):
        losses = None
        reached_eof = False
        current_index = 0
        while not reached_eof:
            batch_x, batch_y, batch_z, actual_batch_size, reached_eof = read_batch_from_samples(dev_samples, batch_size, current_index)
            if actual_batch_size == 0: break
            batch_losses = model.test_model_on_batch(batch_x, batch_y, batch_z)
            if losses is None:
                losses = [0 for _ in batch_losses]
            for i in range(len(batch_losses)):
                losses[i] += (batch_losses[i] * actual_batch_size)
            current_index += actual_batch_size
        for i in range(len(losses)):
            losses[i] /= float(current_index)
        return losses


    @staticmethod
    def get_wsd_metrics(batch_size, dev_samples, model):
        good = 0
        total = 0
        reached_eof = False
        current_index = 0
        while not reached_eof:
            batch_x, batch_y, batch_z, actual_batch_size, reached_eof = read_batch_from_samples(dev_samples, batch_size, current_index)
            if actual_batch_size == 0: break
            output = model.predict_model_on_batch(batch_x)
            for i in range(len(output)):
                for j in range(len(output[i])):
                    if j < len(batch_z[0][i]):
                        restricted_possibilities = batch_z[0][i][j]
                        max_proba = None
                        max_possibility = None
                        for possibility in restricted_possibilities:
                            if possibility != 0:
                                proba = output[i][j][possibility]
                                if max_proba is None or proba > max_proba:
                                    max_proba = proba
                                    max_possibility = possibility
                        if max_possibility is not None:
                            total += 1
                            if max_possibility == batch_y[0][i][j]:
                                good += 1
            current_index += actual_batch_size
        return (float(good) / float(total)) * float(100)
