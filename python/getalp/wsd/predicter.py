from getalp.wsd.common import *
from getalp.wsd.model import Model, ModelConfig
import numpy as np
import sys


class Predicter(object):

    def __init__(self):
        self.training_root_path: str = str()
        self.ensemble_weights_path: str = str()

    def predict(self):
        config_file_path = self.training_root_path + "/config.json"
        config = ModelConfig()
        config.load_from_file(config_file_path)

        model = Model()
        model.config = config

        ensemble = self.create_ensemble(len(self.ensemble_weights_path), model)
        self.load_ensemble_weights(ensemble, self.ensemble_weights_path)

        output = None
        i = 0
        for line in sys.stdin:
            if i == 0:
                sample_x = read_sample_x_or_y_from_string(line)
                output = self.predict_ensemble_on_sample(ensemble, sample_x)
                i = 1
            elif i == 1:
                sample_z = read_sample_z_from_string(line)
                sample_y = self.generate_wsd_on_sample(output, sample_z)
                sys.stdout.write(sample_y + "\n")
                sys.stdout.flush()
                i = 0


    @staticmethod
    def create_ensemble(ensemble_size, model):
        ensemble = []
        for _ in range(ensemble_size):
            copy = Model()
            copy.config = model.config
            copy.create_model()
            ensemble.append(copy)
        return ensemble


    @staticmethod
    def load_ensemble_weights(ensemble, ensemble_weights_paths):
        for i in range(0, len(ensemble)):
            ensemble[i].load_model_weights(ensemble_weights_paths[i])


    @staticmethod
    def predict_ensemble_on_sample(ensemble, sample_x):
        if len(ensemble) == 1:
            return ensemble[0].predict_model_on_sample(sample_x)
        ensemble_sample_y = None
        for model in ensemble:
            model_sample_y = model.predict_model_on_sample(sample_x)
            model_sample_y = np.log(model_sample_y)
            if ensemble_sample_y is None:
                ensemble_sample_y = model_sample_y
            else:
                ensemble_sample_y = np.sum([ensemble_sample_y, model_sample_y], axis=0)
        ensemble_sample_y = np.divide(ensemble_sample_y, len(ensemble))
        ensemble_sample_y = np.exp(ensemble_sample_y)
        return ensemble_sample_y


    @staticmethod
    def generate_wsd_on_sample(output, sample_z):
        sample_y = ""
        for j in range(len(output)):
            if j < len(sample_z[0]):
                restricted_possibilities = sample_z[0][j]
                max_proba = 0
                max_possibility = None
                for possibility in restricted_possibilities:
                    if possibility != 0:
                        proba = output[j][possibility]
                        if proba > max_proba:
                            max_proba = proba
                            max_possibility = possibility
                if max_possibility is not None:
                    sample_y += str(max_possibility) + " "
                else:
                    sample_y += "0 "
        return sample_y

