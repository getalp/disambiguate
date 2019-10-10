from getalp.wsd.common import *
from getalp.wsd.model import Model, ModelConfig, DataConfig
from torch.nn.functional import log_softmax
import sys


class Predicter(object):

    def __init__(self):
        self.training_root_path: str = str()
        self.ensemble_weights_path: List[str] = []
        self.clear_text: bool = bool()
        self.batch_size: int = int()
        self.disambiguate: bool = bool()
        self.translate: bool = False
        self.beam_size: int = int()
        self.output_all_features: bool = bool()
        self.write_log: bool = bool()
        self.data_config: DataConfig = None
        self.log = open('output.log', 'w+')

    def predict(self):
        config_file_path = self.training_root_path + "/config.json"
        self.data_config = DataConfig()
        self.data_config.load_from_file(config_file_path)
        config = ModelConfig(self.data_config)
        config.load_from_file(config_file_path)
        if self.clear_text:
            config.data_config.input_clear_text = [True for _ in range(config.data_config.input_features)]
        if self.data_config.output_features <= 0:
            self.disambiguate = False
        if self.data_config.output_translations <= 0:
            self.translate = False

        assert(self.disambiguate or self.translate)

        ensemble = self.create_ensemble(config, self.ensemble_weights_path)

        i = 0
        batch_x = None
        batch_z = None
        for line in sys.stdin:
            if self.write_log:
                self.log.write("input: " + line)
            if i == 0:
                sample_x = read_sample_x_from_string(line, feature_count=config.data_config.input_features, clear_text=config.data_config.input_clear_text)
                self.preprocess_sample_x(ensemble, sample_x)
                if batch_x is None:
                    batch_x = [[] for _ in range(len(sample_x))]
                for j in range(len(sample_x)):
                    batch_x[j].append(sample_x[j])
                if self.disambiguate and not self.output_all_features:
                    i = 1
                else:
                    if len(batch_x[0]) >= self.batch_size:
                        self.predict_and_output(ensemble, batch_x, batch_z, self.data_config.input_clear_text)
                        batch_x = None
            elif i == 1:
                sample_z = read_sample_z_from_string(line, feature_count=config.data_config.output_features)
                if batch_z is None:
                    batch_z = [[] for _ in range(len(sample_z))]
                for j in range(len(sample_z)):
                    batch_z[j].append(sample_z[j])
                i = 0
                if len(batch_z[0]) >= self.batch_size:
                    self.predict_and_output(ensemble, batch_x, batch_z, self.data_config.input_clear_text)
                    batch_x = None
                    batch_z = None

        if batch_x is not None:
            self.predict_and_output(ensemble, batch_x, batch_z, self.data_config.input_clear_text)

    def create_ensemble(self, config: ModelConfig, ensemble_weights_paths: List[str]):
        ensemble = [Model(config) for _ in range(len(ensemble_weights_paths))]
        for i in range(len(ensemble)):
            ensemble[i].load_model_weights(ensemble_weights_paths[i])
            ensemble[i].set_beam_size(self.beam_size)
        return ensemble

    @staticmethod
    def preprocess_sample_x(ensemble: List[Model], sample_x):
        ensemble[0].preprocess_samples([[sample_x]])

    def _write(self, val):
        if self.write_log:
            self.log.write("output: " + val)
        sys.stdout.write(val)

    def predict_and_output(self, ensemble: List[Model], batch_x, batch_z, clear_text):
        pad_batch_x(batch_x, clear_text)
        output_wsd, output_translation = None, None
        # TODO: refact this horror
        if self.disambiguate and not self.translate and self.output_all_features:
            output_all_features = Predicter.predict_ensemble_all_features_on_batch(ensemble, batch_x)
            batch_all_features = Predicter.generate_all_features_on_batch(output_all_features, batch_x)
            for sample_all_features in batch_all_features:
                self._write(sample_all_features + "\n")
            sys.stdout.flush()
            return
        if self.disambiguate and not self.translate:
            output_wsd = Predicter.predict_ensemble_wsd_on_batch(ensemble, batch_x)
        elif self.translate and not self.disambiguate:
            output_translation = Predicter.predict_ensemble_translation_on_batch(ensemble, batch_x)
        else:
            output_wsd, output_translation = Predicter.predict_ensemble_wsd_and_translation_on_batch(ensemble, batch_x)
        if output_wsd is not None and output_translation is None:
            batch_wsd = Predicter.generate_wsd_on_batch(output_wsd, batch_z)
            for sample_wsd in batch_wsd:
                self._write(sample_wsd + "\n")
        elif output_translation is not None and output_wsd is None:
            batch_translation = Predicter.generate_translation_on_batch(output_translation, ensemble[0].config.data_config.output_translation_vocabularies[0][0])
            for sample_translation in batch_translation:
                self._write(sample_translation + "\n")
        elif output_wsd is not None and output_translation is not None:
            batch_wsd = Predicter.generate_wsd_on_batch(output_wsd, batch_z)
            batch_translation = Predicter.generate_translation_on_batch(output_translation, ensemble[0].config.data_config.output_translation_vocabularies[0][0])
            assert len(batch_wsd) == len(batch_translation)
            for wsd, trans in zip(batch_wsd, batch_translation):
                self._write(wsd + "\n" + trans + "\n")
        sys.stdout.flush()

    @staticmethod
    def predict_ensemble_wsd_on_batch(ensemble: List[Model], batch_x):
        if len(ensemble) == 1:
            return ensemble[0].predict_wsd_on_batch(batch_x)
        ensemble_sample_y = None
        for model in ensemble:
            model_sample_y = model.predict_wsd_on_batch(batch_x)
            model_sample_y = log_softmax(model_sample_y, dim=2)
            if ensemble_sample_y is None:
                ensemble_sample_y = model_sample_y
            else:
                ensemble_sample_y = model_sample_y + ensemble_sample_y
        return ensemble_sample_y

    @staticmethod
    def predict_ensemble_all_features_on_batch(ensemble: List[Model], batch_x):
        if len(ensemble) == 1:
            return ensemble[0].predict_all_features_on_batch(batch_x)
        else:
            # TODO: manage ensemble
            return None

    @staticmethod
    def predict_ensemble_translation_on_batch(ensemble: List[Model], batch_x):
        if len(ensemble) == 1:
            return ensemble[0].predict_translation_on_batch(batch_x)
        else:
            # TODO: manage ensemble
            return None

    @staticmethod
    def predict_ensemble_wsd_and_translation_on_batch(ensemble: List[Model], batch_x):
        if len(ensemble) == 1:
            return ensemble[0].predict_wsd_and_translation_on_batch(batch_x)
        else:
            # TODO: manage ensemble
            return None

    @staticmethod
    def generate_wsd_on_batch(output, batch_z):
        batch_wsd = []
        for i in range(len(batch_z[0])):
            batch_wsd.append(Predicter.generate_wsd_on_sample(output[i], batch_z[0][i]))
        return batch_wsd

    @staticmethod
    def generate_all_features_on_batch(output, batch_x):
        batch_wsd = []
        for i in range(len(batch_x[0])):
            batch_wsd.append(Predicter.generate_all_features_on_sample(output, batch_x, i))
        return batch_wsd

    @staticmethod
    def generate_translation_on_batch(output, vocabulary):
        return unpad_turn_to_text_and_remove_bpe_of_batch_t(output, vocabulary)

    @staticmethod
    def generate_wsd_on_sample(output, sample_z):
        sample_wsd: List[str] = []
        for i in range(len(sample_z)):
            restricted_possibilities = sample_z[i]
            if 0 in restricted_possibilities:
                sample_wsd.append("0")
            elif -1 in restricted_possibilities:
                sample_wsd.append(str(torch_argmax(output[i]).item()))
            else:
                max_proba = None
                max_possibility = None
                for possibility in restricted_possibilities:
                    proba = output[i][possibility]
                    if max_proba is None or proba > max_proba:
                        max_proba = proba
                        max_possibility = possibility
                sample_wsd.append(str(max_possibility))
        return " ".join(sample_wsd)

    @staticmethod
    def generate_all_features_on_sample(output, batch_x, i):
        return " ".join(["/".join([str(torch_argmax(output[k][i][j]).item()) for k in range(len(output))]) for j in range(len(batch_x[0][i]))])
