import glob
import re
import random
from typing import List

import pandas as pd
import numpy as np
import tensorflow as tf


def create_station_rows(file_path: str) -> List[str]:
    return pd.read_csv(
        file_path, encoding="iso-8859-1", dtype=str
    ).station_id.values.tolist()


def get_hour(file_path: bytes) -> int:
    expr = r"/([0-9]*)/.*.csv"
    return int(re.search(expr, file_path.decode("utf-8"))[1])


def load_matrix(file_path: bytes, all_rows: List[str]) -> np.ndarray:
    file_path = file_path.decode("utf-8")
    df = pd.read_csv(file_path, dtype={"start_station_id": str}).set_index(
        "start_station_id", drop=True
    )
    all_rows_set = set(all_rows)
    missing_cols = all_rows_set - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    missing_rows = all_rows_set - set(df.index)
    zero_df = pd.DataFrame({}, index=missing_rows)
    df = df.append(zero_df).fillna(0)
    return df.loc[all_rows, all_rows].values


def data_generator(file_list: List[bytes], all_rows: List[str]):
    for file in file_list:
        matrix = load_matrix(file, all_rows)
        hour_vector = np.zeros(24)
        hour_vector[get_hour(file)] = 1
        yield matrix, hour_vector


def get_dataset(file_names: List[str], stations_file: str) -> tf.data.Dataset:
    all_rows = create_station_rows(stations_file)
    random.shuffle(file_names)
    return tf.data.Dataset.from_generator(
        data_generator,
        (tf.int32, tf.int32),
        (tf.TensorShape([len(all_rows), len(all_rows)]), tf.TensorShape([24])),
        args=[file_names, all_rows],
    )


# Test tf.data.Dataset
if __name__ == "__main__":
    matrix_paths = glob.glob("../data/data/*/2016-*-*.csv")
    stations_path = "../data/stations_data.csv"
    data = get_dataset(matrix_paths, stations_path)
    # Get first 10 samples
    print(data.batch(10).as_numpy_iterator().__next__())
