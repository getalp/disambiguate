import pathlib
import sys
import os


def create_directory_if_not_exists(directory_path):
    pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_abs_path(path):
    if path is None:
        return None
    elif os.path.isabs(path):
        return path
    else:
        return os.path.abspath(path)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise RuntimeError('Boolean value expected.')


def set_if_not_none(value, value_if_none):
    return value_if_none if value is None else value


def get_value_as_str_list(value):
    if value is None:
        return []
    elif isinstance(value, str):
        return [value]
    else:
        return value


def get_value_as_int_list(value):
    if value is None:
        return []
    elif isinstance(value, int):
        return [value]
    else:
        return value


def get_value_as_bool_list(value):
    if value is None:
        return []
    elif isinstance(value, bool):
        return [value]
    else:
        return value


def pad_list(list_to_pad, pad_length, pad_value):
    for i in range(len(list_to_pad), pad_length):
        list_to_pad.append(pad_value)


def count_lines(file_path):
    file = open(file_path)
    line_count = 0
    for _ in file:
        line_count += 1
    file.close()
    return line_count
