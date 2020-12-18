import glob
import pickle
from timeit import timeit
from typing import Tuple, List, Dict
import random

from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import tensorflow as tf


def get_edges_features_from_sparse_matrix(
    sparse_matrix: coo_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_features = np.expand_dims(sparse_matrix.data, axis=1)
    return edge_features, sparse_matrix.row, sparse_matrix.col


def normalize_array(array: np.ndarray) -> np.ndarray:
    tmp = array - array.mean()
    return tmp / np.absolute(tmp).max()


def create_node_features(stations_path: str) -> np.ndarray:
    df = pd.read_csv(
        stations_path,
        encoding="iso-8859-1",
    )
    lon = normalize_array(df.lon.values)
    lat = normalize_array(df.lat.values)
    return np.concatenate(
        (np.expand_dims(lon, axis=1), np.expand_dims(lat, axis=1)), axis=1
    )


def data_generator(file_list: List[bytes], stations_path: bytes):
    node_features = create_node_features(stations_path.decode("utf-8"))

    for file_path in file_list:
        with open(file_path, "rb") as file:
            feature_list: List[Dict] = pickle.load(file)
        for feature in feature_list:
            edge_features = get_edges_features_from_sparse_matrix(
                feature["current_rides"]
            )
            graph_tuple = (node_features, *edge_features, np.array([0]))
            label_dense = np.zeros(24)
            label_dense[feature["start_time"].hour] = 1
            yield graph_tuple, label_dense


def get_dataset(file_names: List[str], stations_path: str) -> tf.data.Dataset:
    random.shuffle(file_names)
    return tf.data.Dataset.from_generator(
        data_generator,
        ((tf.float64, tf.float64, tf.int32, tf.int32, tf.float64), tf.int32),
        (
            (
                tf.TensorShape([None, 2]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([1]),
            ),
            tf.TensorShape([24]),
        ),
        args=[file_names, stations_path],
    )


if __name__ == "__main__":
    matrix_paths = glob.glob("../data/pickle_data/*.pickle")
    data = get_dataset(matrix_paths, "../data/stations_data.csv")
    # time performance
    iterator = data.batch(1).as_numpy_iterator()
    iterator.__next__()
    print(timeit(lambda: [x for x in iterator], number=2))
