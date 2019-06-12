import json
from typing import List
from getalp.common.common import eprint, count_lines
from getalp.wsd.torch_fix import *
from getalp.wsd.torch_utils import cpu_device
from torch.nn.utils.rnn import pad_sequence

pad_token_index = 0
unk_token_index = 1
bos_token_index = 2
eos_token_index = 3
reserved_token_count = 4
pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"


def get_vocabulary(vocabulary_file_path):
    vocabulary_file = open(vocabulary_file_path)
    vocabulary = [line.rstrip().split()[-1] for line in vocabulary_file]
    vocabulary_file.close()
    return vocabulary


def get_vocabulary_size(vocabulary_file_path):
    return count_lines(vocabulary_file_path)


def get_embeddings_size(embeddings_file_path):
    f = open(embeddings_file_path)
    line = f.readline()
    embeddings_size = len(line.split()[1:])
    f.close()
    return embeddings_size


def get_pretrained_embeddings(pretrained_model_path):
    embeddings_count = reserved_token_count + get_vocabulary_size(pretrained_model_path)
    embeddings_size = get_embeddings_size(pretrained_model_path)
    embeddings = torch_empty((embeddings_count, embeddings_size), dtype=torch_float32, device=cpu_device)
    embeddings[pad_token_index] = torch_zeros(embeddings_size)  # <pad> = 0
    embeddings[unk_token_index] = torch_zeros(embeddings_size)  # <unk> = 1
    embeddings[bos_token_index] = torch_zeros(embeddings_size)  # <bos> = 2
    embeddings[eos_token_index] = torch_zeros(embeddings_size)  # <eos> = 3
    i = reserved_token_count
    f = open(pretrained_model_path)
    for line in f:
        vector = line.split()[1:]
        if len(vector) != embeddings_size:
            eprint("Warning: cannot load pretrained embedding at index " + str(i-reserved_token_count))
            continue
        vector = [float(i) for i in vector]
        embeddings[i] = torch_tensor(vector, dtype=torch_float32, device=cpu_device)
        i += 1
    f.close()
    return embeddings


def read_sample_x_from_string(string: str, feature_count: int, clear_text: List[bool]):
    sample_x: List = [[] for _ in range(feature_count)]
    for word in string.split():
        word_features = word.split('/')
        for i in range(feature_count):
            if clear_text[i]:
                sample_x[i].append(word_features[i].replace("<slash>", "/"))
            else:
                sample_x[i].append(int(word_features[i]))
    for i in range(feature_count):
        if not clear_text[i]:
            sample_x[i] = torch_tensor(sample_x[i], dtype=torch_long, device=cpu_device)
    return sample_x


def read_sample_y_from_string(string: str, feature_count: int):
    sample_y: List = [[] for _ in range(feature_count)]
    for word in string.split():
        word_features = word.split('/')
        for i in range(feature_count):
            sample_y[i].append(int(word_features[i]))
    for i in range(feature_count):
        sample_y[i] = torch_tensor(sample_y[i], dtype=torch_long, device=cpu_device)
    return sample_y


def read_sample_z_from_string(string: str, feature_count: int):
    sample_z = [[] for _ in range(feature_count)]
    for word in string.split():
        word_features = word.split('/')
        for i in range(feature_count):
            sample_z[i].append([int(j) for j in word_features[i].split(";")])
    return sample_z


def read_sample_t_from_string(string: str, feature_count: int):
    sample_t: List = [[] for _ in range(feature_count)]
    for word in string.split():
        word_features = word.split('/')
        for i in range(feature_count):
            sample_t[i].append(int(word_features[i]))
    for i in range(feature_count):
        sample_t[i].append(eos_token_index)
        sample_t[i] = torch_tensor(sample_t[i], dtype=torch_long, device=cpu_device)
    return sample_t


def read_samples_from_file(file_path: str, input_clear_text: List[bool], output_features: int, output_translations: int, output_translation_features: int, output_translation_clear_text: bool, limit: int = -1):
    file = open(file_path, "r")
    samples = []
    sample_triplet = []
    sample_tt = []
    i = 0
    for line in file:
        if i == 0:
            if limit > 0 and len(samples) >= limit > 0:
                break
            sample_x = read_sample_x_from_string(line, feature_count=len(input_clear_text), clear_text=input_clear_text)
            sample_triplet = [sample_x]
            if output_features > 0:
                i = 1
            else:
                sample_triplet.append([])
                sample_triplet.append([])
                i = 3
        elif i == 1:
            sample_y = read_sample_y_from_string(line, feature_count=output_features)
            sample_triplet.append(sample_y)
            i = 2
        elif i == 2:
            sample_z = read_sample_z_from_string(line, feature_count=output_features)
            sample_triplet.append(sample_z)
            if output_translations > 0:
                i = 3
            else:
                sample_triplet.append([])
                samples.append(sample_triplet)
                i = 0
        elif i == 3:
            if output_translation_clear_text:
                raise NotImplementedError
                # TODO:
                # sample_t = read_sample_x_from_string(line, feature_count=0, clear_text=[True], add_eos_token=False)
            else:
                sample_t = read_sample_t_from_string(line, feature_count=output_translation_features)
            sample_tt.append(sample_t)
            if len(sample_tt) >= output_translations:
                sample_triplet.append(sample_tt)
                sample_tt = []
                samples.append(sample_triplet)
                i = 0
    file.close()
    return samples


def pad_batch_x(batch_x, clear_text):
    for i in range(len(batch_x)):
        if not clear_text[i]:
            batch_x[i] = pad_sequence(batch_x[i], batch_first=True)


def pad_batch_y(batch_y):
    for i in range(len(batch_y)):
        batch_y[i] = pad_sequence(batch_y[i], batch_first=True)


def pad_batch_tt(batch_tt):
    for i in range(len(batch_tt)):
        for j in range(len(batch_tt[i])):
            batch_tt[i][j] = pad_sequence(batch_tt[i][j], batch_first=True)


def unpad_turn_to_text_and_remove_bpe_of_batch_t(batch_t, vocabulary: List[str]):
    ret: List[str] = []
    for k in range(len(batch_t)):
        str_as_list = []
        for l in range(len(batch_t[k])):
            value = batch_t[k][l].item()
            if value == eos_token_index or value == pad_token_index:
                break
            value = vocabulary[value]
            str_as_list.append(value)
        str_as_str = " ".join(str_as_list)
        str_as_str = str_as_str.replace("@@ ", "")
        str_as_str = str_as_str.replace(" ##", "")
        ret.append(str_as_str)
    return ret


def read_batch_from_samples(samples, batch_size: int, token_per_batch: int, current_index: int, input_features: int, output_features: int, output_translations: int, output_translation_features: int, input_clear_text: List[bool], output_translation_clear_text: bool):
    batch_x = [[] for _ in range(input_features)]
    batch_y = [[] for _ in range(output_features)]
    batch_z = [[] for _ in range(output_features)]
    batch_tt = [[[] for __ in range(output_translation_features)] for _ in range(output_translations)]
    actual_batch_size = 0
    reached_eof = False
    max_length = 0
    max_length_tt: List[int] = [0 for _ in range(output_translations)]

    while True:
        if current_index >= len(samples):
            reached_eof = True
            break
        if actual_batch_size >= batch_size > 0:
            break

        sample = samples[current_index]

        max_length_if_accepted = max(max_length, len(sample[0][0]))
        if (actual_batch_size + 1) * max_length_if_accepted > token_per_batch > 0:
            break
        max_length = max_length_if_accepted

        sample_x = sample[0]
        for j in range(input_features):
            batch_x[j].append(sample_x[j])

        sample_y = sample[1]
        for j in range(output_features):
            batch_y[j].append(sample_y[j])

        sample_z = sample[2]
        for j in range(output_features):
            batch_z[j].append(sample_z[j])

        sample_tt = sample[3]
        for j in range(output_translations):
            for k in range(output_translation_features):
                batch_tt[j][k].append(sample_tt[j][k])
            max_length_tt[j] = max(max_length_tt[j], len(sample_tt[j][0]))

        actual_batch_size += 1
        current_index += 1

    pad_batch_x(batch_x, input_clear_text)
    pad_batch_y(batch_y)
    pad_batch_tt(batch_tt)  # TODO: output_translation_clear_text

    return batch_x, batch_y, batch_z, batch_tt, actual_batch_size, reached_eof


def save_training_info(file_path, current_ensemble, current_epoch, current_batch, current_batch_total, train_line, current_best_loss, current_best_wsd, current_best_bleu, random_seed):
    info = {"current_ensemble": current_ensemble,
            "current_epoch": current_epoch,
            "current_batch": current_batch,
            "current_batch_total": current_batch_total,
            "train_line": train_line,
            "current_best_loss": current_best_loss,
            "current_best_wsd": current_best_wsd,
            "current_best_bleu": current_best_bleu,
            "random_seed": random_seed
            }
    file = open(file_path, "w")
    json.dump(info, file)
    file.close()


def load_training_info(file_path):
    file = open(file_path, "r")
    info = json.load(file)
    file.close()
    return (info["current_ensemble"],
            info["current_epoch"],
            info["current_batch"],
            info["current_batch_total"],
            info["train_line"],
            info["current_best_loss"],
            info["current_best_wsd"],
            info["current_best_bleu"],
            info["random_seed"]
            )
