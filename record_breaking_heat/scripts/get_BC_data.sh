#!/bin/bash

# Pull data from Environment Canada
# These are stations in BC that have data from 1925-2021
# Even though the command has a spot for month and day, only year matters

for station_id in 568 707 1032 1039 1142 1180 1340 1364
    do
    for yy in {1925..2021}
        do
        wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=${station_id}&Year=${yy}&Month=1&Day=14&timeframe=2&submit= Download+Data"
    done
done
