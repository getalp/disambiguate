import random
import numpy as np
import json
from typing import List


def get_vocabulary_size(vocabulary_file_path):
    vocabulary_size = 0
    vocabulary_file = open(vocabulary_file_path)
    for _ in vocabulary_file:
        vocabulary_size += 1
    vocabulary_file.close()
    return vocabulary_size


def get_embeddings_size(embeddings_file_path):
    f = open(embeddings_file_path)
    line = f.readline()
    embeddings_size = len(line.split()[1:])
    f.close()
    return embeddings_size


def load_vocabulary(vocabulary_file_path):
    vocabulary_file = open(vocabulary_file_path)
    vocabulary = []
    for line in vocabulary_file:
        vocabulary.append(line.split()[1])
    vocabulary_file.close()
    return vocabulary


def get_pretrained_embeddings(pretrained_model_path):
    embeddings_count = 2 + get_vocabulary_size(pretrained_model_path)
    embeddings_size = get_embeddings_size(pretrained_model_path)
    embeddings = np.empty(shape=(embeddings_count, embeddings_size), dtype=np.float32)
    embeddings[0] = np.zeros(embeddings_size)  # <padding> = 0
    embeddings[1] = np.zeros(embeddings_size)  # <unknown> = 1
    i = 2
    f = open(pretrained_model_path)
    for line in f:
        vector = line.split()[1:]
        vector = [float(i) for i in vector]
        embeddings[i] = np.array(vector, dtype=np.float32)
        i += 1
    f.close()
    return embeddings


def read_sample_x_or_y_from_string(string):
    sample_x: List = None
    for word in string.split():
        word_features = word.split('/')
        if sample_x is None:
            sample_x = [[] for _ in range(len(word_features))]
        for i in range(len(word_features)):
            sample_x[i].append(int(word_features[i]))
    for i in range(len(sample_x)):
        sample_x[i] = np.array(sample_x[i], dtype=np.int64)
    return sample_x


def read_sample_z_from_string(string):
    sample_y = None
    for word in string.split():
        word_features = word.split('/')
        if sample_y is None:
            sample_y = [[] for _ in range(len(word_features))]
        for i in range(len(word_features)):
            sample_y[i].append([int(j) for j in word_features[i].split(";")])
    return sample_y


def read_all_samples_from_file(file_path):
    file = open(file_path, "r")
    samples = []
    sample_triplet = []
    i = 0
    for line in file:
        if i == 0:
            sample_x = read_sample_x_or_y_from_string(line)
            sample_triplet = [sample_x]
            i = 1
        elif i == 1:
            sample_y = read_sample_x_or_y_from_string(line)
            sample_triplet.append(sample_y)
            i = 2
        elif i == 2:
            sample_z = read_sample_z_from_string(line)
            sample_triplet.append(sample_z)
            samples.append(sample_triplet)
            i = 0
    file.close()
    return samples


def create_fake_batch(batch_size, sample_size, input_features, input_vocabulary_sizes, output_features, output_vocabulary_sizes):
    batch_x = []
    for i in range(input_features):
        feature_batch_x = []
        for j in range(batch_size):
            sample_x = []
            for k in range(sample_size):
                sample_x.append(random.randrange(0, input_vocabulary_sizes[i]))
            feature_batch_x.append(sample_x)
        feature_batch_x = np.array(feature_batch_x, dtype=np.int64)
        batch_x.append(feature_batch_x)
    batch_y = []
    for i in range(output_features):
        feature_batch_y = []
        for j in range(batch_size):
            sample_y = []
            for k in range(sample_size):
                sample_y.append(random.randrange(0, output_vocabulary_sizes[i]))
            feature_batch_y.append(sample_y)
        feature_batch_y = np.array(feature_batch_y, dtype=np.int64)
        batch_y.append(feature_batch_y)
    return batch_x, batch_y


def read_batch_from_samples(samples, batch_size, current_index):
    batch_x = None
    batch_y = None
    batch_z = None
    actual_batch_size = 0
    reached_eof = False
    max_length = 0

    for i in range(current_index, current_index + batch_size):
        if i >= len(samples):
            reached_eof = True
            break

        sample = samples[i]

        sample_x = sample[0]
        if batch_x is None:
            batch_x = [[] for _ in range(len(sample_x))]
        for j in range(len(sample_x)):
            batch_x[j].append(sample_x[j])
        max_length = max(max_length, len(sample_x[0]))

        sample_y = sample[1]
        if batch_y is None:
            batch_y = [[] for _ in range(len(sample_y))]
        for j in range(len(sample_y)):
            batch_y[j].append(sample_y[j])

        sample_z = sample[2]
        if batch_z is None:
            batch_z = [[] for _ in range(len(sample_z))]
        for j in range(len(sample_z)):
            batch_z[j].append(sample_z[j])

        actual_batch_size += 1

    for j in range(0, actual_batch_size):
        padding_needed = max_length - len(batch_x[0][j])
        for i in range(len(batch_x)):
            batch_x[i][j] = np.pad(batch_x[i][j], (0, padding_needed), mode='constant', constant_values=0)
        for i in range(len(batch_y)):
            batch_y[i][j] = np.pad(batch_y[i][j], (0, padding_needed), mode='constant', constant_values=0)

    if batch_x is None:
        batch_x = []

    if batch_y is None:
        batch_y = []

    if batch_z is None:
        batch_z = []

    for i in range(len(batch_x)):
        batch_x[i] = np.array(batch_x[i], dtype=np.int64)

    for i in range(len(batch_y)):
        batch_y[i] = np.array(batch_y[i], dtype=np.int64)

    return batch_x, batch_y, batch_z, actual_batch_size, reached_eof


def save_training_info(file_path, current_ensemble, current_epoch, current_batch, train_line, current_best_wsd, current_best_loss):
    info = {"current_ensemble":current_ensemble,
            "current_epoch":current_epoch,
            "current_batch":current_batch,
            "train_line":train_line,
            "current_best_wsd":current_best_wsd,
            "current_best_loss":current_best_loss,
            }
    file = open(file_path, "w")
    json.dump(info, file)
    file.close()


def load_training_info(file_path):
    file = open(file_path, "r")
    info = json.load(file)
    file.close()
    return info["current_ensemble"], info["current_epoch"], info["current_batch"], info["train_line"], info["current_best_wsd"], info["current_best_loss"]


def save_training_losses(file_path, train_loss, dev_loss, dev_wsd):
    file = open(file_path, "a")
    file.write(str(train_loss) + " " + str(dev_loss) + " " + str(dev_wsd) + "\n")
    file.close()


def load_training_losses(file_path):
    file = open(file_path, "r")
    train_losses = []
    dev_losses = []
    dev_wsd = []
    for line in file:
        linesplit = line.split()
        train_losses.append(float(linesplit[0]))
        dev_losses.append(float(linesplit[1]))
        dev_wsd.append(float(linesplit[2]))
    file.close()
    return train_losses, dev_losses, dev_wsd
