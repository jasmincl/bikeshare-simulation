# bikeshare-simulation
Simulation and visualization of bikesharing station trip attraction &amp; production.





## Data
The main data source is a dataset of bikesharing trips with  start and stop station
in Hamburg from "Call A Bike", a station based bikesharing system from Deutsche Bahn AG. The dataset is
available [here](https://data.deutschebahn.com/dataset/data-call-a-bike "Deutsche Bahn Open Data").

### Stations data
File: `data/stations_data.csv`

All 208 bikeshare stations of "StadtRad" or "Call A Bike" Hamburg. For each station the file contains name, unique station_id, latitude and longitude. 

### Trip data
The trip data contains 8,265,966 rows of bikesharing trips between January 2014 and May 2017. 

The following columns are in the data set: 
- booking, vehicle and customer id
- start time & date of booking and end time & date (`datetime_from`and `datetime_to`)
- `start_station_id` and `end_station_id`
- `hour_from` and `hour_to` (hour of rental ranging from 0 to 23)
- `duration_trip_minutes`. How much time passed between rental at start station and return at end station.

Notes: Loop trips (e.g. start and stop station are the same) are possible. 