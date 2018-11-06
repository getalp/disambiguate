import pathlib


def create_directory_if_not_exists(directory_path):
    pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)

