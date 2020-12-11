import glob
import pickle
import random
from typing import List

import numpy as np
import tensorflow as tf


def data_generator(file_list: List[bytes]):
    for file_path in file_list:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        for feature, label in zip(data["feature"], data["label"]):
            label_dense = np.zeros(24)
            label_dense[label] = 1
            yield feature.toarray(), label_dense


def get_dataset(file_names: List[str], number_stations: int) -> tf.data.Dataset:
    random.shuffle(file_names)
    return tf.data.Dataset.from_generator(
        data_generator,
        (tf.int32, tf.int32),
        (tf.TensorShape([number_stations, number_stations]), tf.TensorShape([24])),
        args=[file_names])


# Test tf.data.Dataset
if __name__ == "__main__":
    matrix_paths = glob.glob("../data/pickle_data/*.pickle")
    data = get_dataset(matrix_paths, 208)
    # Get first 10 samples
    print(data.batch(10).as_numpy_iterator().__next__())

