# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:13:32 2020

@author: freddy
Create a JSON file, containing date, time and corresponding weight matrix.

"""

import pandas as pd
import networkx as nx
import numpy as np
import json
data = pd.read_csv("../data/biketrip_data.csv")

data_agg = data[["start_station_id", "end_station_id","hour_from","date_from" ]]
data_agg["n"] = 1
data_agg_detail = data_agg.groupby(["start_station_id","end_station_id","date_from", "hour_from"]).sum()
data_agg_detail = data_agg_detail.reset_index()

results = dict()
for date in data_agg_detail["date_from"].unique():  # for every day
    filtered_data = data_agg_detail[data_agg_detail["date_from"]==date]  # filter date
    for time in filtered_data["hour_from"].unique():  # for every time
        filtered_data_time = filtered_data[filtered_data["hour_from"]==time]  # filter time
        date_var = filtered_data_time["date_from"].unique().tolist()[0]  # save date
        time_var = filtered_data_time["hour_from"].unique().tolist()[0]  # save time
        # create the matrix
        matrix = pd.crosstab(index=filtered_data_time["start_station_id"], columns=filtered_data_time["end_station_id"], values = filtered_data_time["n"], aggfunc="sum")
        if date_var in results.keys():  # if day already exists
            results[date_var][time_var] = matrix.to_json()
        else:
            results[date_var] = dict()
            results[date_var][time_var] = matrix.to_json()
# save as .json
with open("../data/bikeshare_agg_matrix.json", "w") as file:
    json.dump(results, file)
        
