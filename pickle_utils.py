import pickle
import gzip


def gload_data(file_path):
    with gzip.GzipFile(file_path, "rb") as file:
        return pickle.load(file)


def gsave_data(obj, file_path):
    with gzip.GzipFile(file_path, "wb") as file:
        pickle.dump(obj, file, -1)


def save_data(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
