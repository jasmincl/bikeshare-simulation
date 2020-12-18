import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import tqdm as tqdm
from scipy.sparse import coo_matrix


def load_stations_mapping(file_path: str) -> Dict[str, int]:
    station_ids = pd.read_csv(
        file_path, encoding="iso-8859-1", dtype=str
    ).station_id.values
    return {val: ind for ind, val in enumerate(station_ids)}


def create_sparse_matrix(
    data: pd.DataFrame,
    min_datetime: datetime,
    max_datetime: datetime,
    stations_mapping: Dict[str, int],
) -> coo_matrix:
    df = data.loc[min_datetime.isoformat() : max_datetime.isoformat()]
    dense_matrix = np.zeros([len(stations_mapping), len(stations_mapping)])
    for row in df[["start_station_id", "end_station_id"]].values:
        dense_matrix[stations_mapping[row[0]], stations_mapping[row[1]]] += 1

    return coo_matrix(dense_matrix)


def load_stations_data(file_path: str) -> pd.DataFrame:
    dtypes = {
        "datetime_from": str,
        "start_station_id": str,
        "end_station_id": str,
        "hour_from": int,
    }
    data = pd.read_csv(file_path, usecols=list(dtypes.keys()), dtype=dtypes)
    data["datetime_from"] = pd.to_datetime(data.datetime_from)
    return data.set_index("datetime_from")


def get_min_max_dateime_array(
    first_timestamp: pd.Timestamp, last_timestamp: pd.Timestamp
) -> List[Tuple[datetime, datetime]]:
    result = []
    min_time = datetime(
        first_timestamp.year,
        first_timestamp.month,
        first_timestamp.day,
        first_timestamp.hour,
    )
    max_time = min_time + timedelta(hours=1)
    while min_time < last_timestamp:
        result.append((min_time, max_time))
        min_time = max_time
        max_time = min_time + timedelta(hours=1)

    return result


def write_pickle_files(
    stations_data: pd.DataFrame, stations_mapping: Dict[str, int], path: str
) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    time_array = get_min_max_dateime_array(
        stations_data.index[0], stations_data.index[-1]
    )

    tmp_hours, tmp_sparse_array = [], []
    for min_datetime, max_datetime in tqdm.tqdm(time_array):
        tmp_sparse_array.append(
            create_sparse_matrix(
                stations_data, min_datetime, max_datetime, stations_mapping
            )
        )
        tmp_hours.append(min_datetime.hour)
        if max_datetime.day == 1 and max_datetime.hour == 0:
            file_name = f"{min_datetime.year}-{min_datetime.month:02}.pickle"
            with open(os.path.join(path, file_name), "wb") as file:
                pickle.dump(
                    {
                        "start_time": min_datetime,
                        "end_time": max_datetime,
                        "current_rides": tmp_sparse_array,
                    },
                    file,
                )
            tmp_hours, tmp_sparse_array = [], []


if __name__ == "__main__":
    g_stations_data = load_stations_data("../data/biketrip_data.csv")
    g_stations_mapping = load_stations_mapping("../data/stations_data.csv")

    write_pickle_files(g_stations_data, g_stations_mapping, "../data/pickle_data")
