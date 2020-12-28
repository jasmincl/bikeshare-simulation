import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import tqdm as tqdm


def create_sparse_matrix(
    data: pd.DataFrame,
    min_datetime: datetime,
    max_datetime: datetime,
) -> Dict[str, np.ndarray]:
    df = data.loc[min_datetime.isoformat() : max_datetime.isoformat()]
    group_by = df.groupby(["start_station_id", "end_station_id"]).duration
    number_rides = group_by.count()
    return {
        "start_station_id": number_rides.index.get_level_values(0),
        "end_station_id": number_rides.index.get_level_values(1),
        "number_rides": number_rides.values,
        "mean_duration": group_by.mean(numeric_only=False).values,
    }


def load_stations_data(file_path: str) -> pd.DataFrame:
    dtypes = {
        "datetime_from": str,
        "datetime_to": str,
        "start_station_id": str,
        "end_station_id": str,
        "hour_from": int,
    }
    data = pd.read_csv(file_path, usecols=list(dtypes.keys()), dtype=dtypes)
    data["datetime_from"] = pd.to_datetime(data.datetime_from)
    data["duration"] = pd.to_datetime(data.datetime_to) - data.datetime_from
    return data.set_index("datetime_from")


def get_min_max_datetime_array(
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


def write_pickle_files(stations_data: pd.DataFrame, path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    time_array = get_min_max_datetime_array(
        stations_data.index[0], stations_data.index[-1]
    )

    features, last_rides = [], None
    for min_datetime, max_datetime in tqdm.tqdm(time_array):
        current_rides = create_sparse_matrix(stations_data, min_datetime, max_datetime)
        if last_rides is not None:
            features.append(
                dict(
                    start_time=min_datetime,
                    end_time=max_datetime,
                    current_rides=current_rides,
                    last_rides=last_rides,
                )
            )

        if max_datetime.day == 1 and max_datetime.hour == 0:
            file_name = f"{min_datetime.year}-{min_datetime.month:02}.pickle"
            with open(os.path.join(path, file_name), "wb") as file:
                pickle.dump(features, file)
            features = []
        last_rides = current_rides


if __name__ == "__main__":
    g_stations_data = load_stations_data("../data/biketrip_data.csv")

    write_pickle_files(g_stations_data, "../data/pickle_data")
