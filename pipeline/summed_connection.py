import glob
import pickle
import random
from functools import partial
from timeit import timeit
from typing import List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf


def load_stations_mapping(file_path: str) -> Dict[str, int]:
    station_ids = pd.read_csv(
        file_path, encoding="iso-8859-1", dtype=str
    ).station_id.values
    return {val: ind for ind, val in enumerate(station_ids)}


def get_summed_connections(
    matrix_dict: Dict[str, np.ndarray], station_mapping: Dict[str, int]
) -> np.ndarray:
    result = np.zeros(len(station_mapping))
    for station_id, number_rides in zip(
        matrix_dict["start_station_id"], matrix_dict["number_rides"]
    ):
        result[station_mapping[station_id]] += number_rides

    return result


def data_generator(file_list: List[bytes], station_mapping: Dict[str, int]):
    for file_path in file_list:
        with open(file_path, "rb") as file:
            feature_list: List[Dict] = pickle.load(file)
        for feature in feature_list:
            label_dense = np.zeros(24)
            label_dense[feature["start_time"].hour] = 1
            train_input = get_summed_connections(
                feature["current_rides"], station_mapping
            )
            yield train_input, label_dense


def get_dataset(file_names: List[str], station_file_path: str) -> tf.data.Dataset:
    station_mapping = load_stations_mapping(station_file_path)
    generator = partial(data_generator, station_mapping=station_mapping)
    random.shuffle(file_names)
    return tf.data.Dataset.from_generator(
        generator,
        (tf.int32, tf.int32),
        (tf.TensorShape([len(station_mapping)]), tf.TensorShape([24])),
        args=[file_names],
    )


# Test tf.data.Dataset
if __name__ == "__main__":
    matrix_paths = glob.glob("../data/pickle_data/*.pickle")
    data = get_dataset(matrix_paths, "../data/stations_data.csv")
    # time performance
    iterator = data.batch(10).as_numpy_iterator()
    iterator.__next__()
    print(timeit(lambda: [x for x in iterator], number=2))
