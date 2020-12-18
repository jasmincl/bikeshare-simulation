import glob
import pickle
import random
from timeit import timeit
from typing import List, Dict

import numpy as np
import tensorflow as tf


def data_generator(file_list: List[bytes]):
    for file_path in file_list:
        with open(file_path, "rb") as file:
            feature_list: List[Dict] = pickle.load(file)
        for feature in feature_list:
            label_dense = np.zeros(24)
            label_dense[feature["start_time"].hour] = 1
            yield feature["current_rides"].toarray(), label_dense


def get_dataset(file_names: List[str], number_stations: int) -> tf.data.Dataset:
    random.shuffle(file_names)
    return tf.data.Dataset.from_generator(
        data_generator,
        (tf.int32, tf.int32),
        (tf.TensorShape([number_stations, number_stations]), tf.TensorShape([24])),
        args=[file_names],
    )


# Test tf.data.Dataset
if __name__ == "__main__":
    matrix_paths = glob.glob("../data/pickle_data/*.pickle")
    data = get_dataset(matrix_paths, 208)
    # time performance
    iterator = data.batch(10).as_numpy_iterator()
    iterator.__next__()
    print(timeit(lambda: [x for x in iterator], number=2))
