import glob
import pickle
from datetime import datetime
from functools import partial
from timeit import timeit
from typing import Tuple, List, Dict
import random

import numpy as np
import pandas as pd
import tensorflow as tf


NODE_FEATURE_DIM = 2
EDGE_FEATURE_DIM = 2
GLOBAL_FEATURE_DIM = 7


def get_edge_features(matrix_dict: Dict[str, np.ndarray]) -> np.ndarray:
    number_rides = np.expand_dims(
        matrix_dict["number_rides"].astype(np.float64), axis=1
    )
    mean_duration = np.expand_dims(
        matrix_dict["mean_duration"].astype("timedelta64[m]").astype(np.float64), axis=1
    )
    return np.concatenate([number_rides, mean_duration], axis=1)


def get_nodes_and_edges_features(
    matrix_dict: Dict[str, np.ndarray], node_mapping: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    number_edges = len(matrix_dict["start_station_id"])
    if number_edges == 0:
        return (
            np.zeros([1, NODE_FEATURE_DIM]),
            np.zeros([1, EDGE_FEATURE_DIM]),
            np.array([0]),
            np.array([0]),
        )
    all_stations = [matrix_dict["start_station_id"], matrix_dict["end_station_id"]]
    unique_stations, indices = np.unique(
        np.concatenate(all_stations), return_inverse=True
    )
    node_feature = np.array([node_mapping[x] for x in unique_stations])
    edge_features = get_edge_features(matrix_dict)
    return node_feature, edge_features, indices[:number_edges], indices[number_edges:]


def normalize_array(array: np.ndarray) -> np.ndarray:
    tmp = array - array.mean()
    return tmp / np.absolute(tmp).max()


def create_node_mapping(stations_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(
        stations_path,
        encoding="iso-8859-1",
        dtype={"station_id": str, "lon": np.float64, "lat:": np.float64},
    )
    lon = normalize_array(df.lon.values)
    lat = normalize_array(df.lat.values)
    return dict(zip(df.station_id.values, np.column_stack([lon, lat])))


def create_global_features(date: datetime) -> np.ndarray:
    result = np.zeros(7)
    result[date.weekday()] = 1
    return result


def data_generator(file_list: List[bytes], node_mapping: Dict[str, np.ndarray]):
    for file_path in file_list:
        with open(file_path, "rb") as file:
            feature_list: List[Dict] = pickle.load(file)
        for feature in feature_list:
            local_features = get_nodes_and_edges_features(
                feature["current_rides"], node_mapping
            )
            global_features = create_global_features(feature["start_time"])
            graph_tuple = (*local_features, global_features)
            label_dense = np.zeros(24)
            label_dense[feature["start_time"].hour] = 1
            yield graph_tuple, label_dense


def get_dataset(file_names: List[str], stations_path: str) -> tf.data.Dataset:
    node_mapping = create_node_mapping(stations_path)
    gen = partial(data_generator, node_mapping=node_mapping)
    random.shuffle(file_names)
    return tf.data.Dataset.from_generator(
        gen,
        ((tf.float64, tf.float64, tf.int32, tf.int32, tf.float64), tf.int32),
        (
            (
                tf.TensorShape([None, NODE_FEATURE_DIM]),
                tf.TensorShape([None, EDGE_FEATURE_DIM]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([GLOBAL_FEATURE_DIM]),
            ),
            tf.TensorShape([24]),
        ),
        args=[file_names],
    )


if __name__ == "__main__":
    matrix_paths = glob.glob("../data/pickle_data/*.pickle")
    data = get_dataset(matrix_paths, "../data/stations_data.csv")
    # time performance
    iterator = data.batch(1).as_numpy_iterator()
    iterator.__next__()
    print(timeit(lambda: [x for x in iterator], number=2))
