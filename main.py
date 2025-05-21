import os
import pandas as pd
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta

if os.path.exists("stops_df.csv"):
    stops_df = pd.read_csv("stops_df.csv", parse_dates=["tstamp"])
else:
    with open("trimet_stopevents_2022-12-07.html", "r") as file:
        soup = bs(file, "html.parser")

    tables = soup.find_all("table")

    rows = []
    for table in tables:
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 15:
                continue
            trip_id = tds[6].text.strip()
            vehicle_number = tds[0].text.strip()
            try:
                arrive_time = int(tds[8].text.strip())
                tstamp = datetime(2022, 12, 7) + timedelta(seconds=arrive_time)
            except ValueError:
                continue
            location_id = tds[10].text.strip()
            ons = int(tds[13].text.strip())
            offs = int(tds[14].text.strip())
            rows.append([trip_id, vehicle_number, tstamp, location_id, ons, offs])

    stops_df = pd.DataFrame(rows, columns=["trip_id", "vehicle_number", "tstamp", "location_id", "ons", "offs"])
    stops_df.to_csv("stops_df.csv", index=False)

print(stops_df.head())
print(f"Total stop events: {len(stops_df)}")

print("\n--- Section 2: Transform the Data ---")

# 2. transform the data

# a. how many vehicles are contained in the data
num_vehicles = stops_df['vehicle_number'].nunique()
print("Number of vehicles:", num_vehicles)

# b. how many stop locations
num_locations = stops_df["location_id"].nunique()
print("Number of stop locations:", num_locations)

# c. min and max values of tstamp
min_tstamp = stops_df["tstamp"].min()
max_tstamp = stops_df["tstamp"].max()
print("Min tstamp:", min_tstamp)
print("Max tstamp:", max_tstamp)

# d. stop events which at least one passenger boarded
num_with_boarding = (stops_df["ons"] >= 1).sum()
print("Stop events with at least one boarding:", num_with_boarding)

# e. percentage of stop events with at least one passenger boarding
percent_boarding = (num_with_boarding / len(stops_df)) * 100
print(f"Percentage of stop events with boardings: {percent_boarding:.2f}%")

print("\n--- Section 3A: Location 6913 ---")

# 3. validate

# 3a. location 6913
loc_df = stops_df[stops_df["location_id"] == 6913]
#i. how many stops made at this location
num_stops_6913 = len(loc_df)
print("Stops at location 6913:", num_stops_6913)
# ii. how many different buses stopped at this location
unique_buses_6913 = loc_df["vehicle_number"].nunique()
print("Unique buses at location 6913:", unique_buses_6913)
# iii. percentage of stops with at least one passenger
boarding_stops_6913 = (loc_df["ons"] >= 1).sum()
percentage_boarding_6913 = (boarding_stops_6913 / num_stops_6913) * 100 if num_stops_6913 > 0 else 0
print(f"Percentage of stops with boarding at location 6913: {percentage_boarding_6913:.2f}%")

print("\n--- Section 3B: Vehicle 4062 ---")

# 3b. location 4062 
veh_df = stops_df[stops_df["vehicle_number"] == 4062]

# how many stops made by this vehicle
num_stops_4062 = len(veh_df)
print("Stops by vehicle 4062:", num_stops_4062)

# total passengers boarded
total_ons_4062 = veh_df["ons"].sum()
print("Total passengers boarded on vehicle 4062:", total_ons_4062)

# total passangers deboarded
total_offs_4062 = veh_df["offs"].sum()
print("Total passengers deboarded from vehicle 4062:", total_offs_4062)

# percentage of stops where at least one passenger boarded
boarding_stops_4062 = (veh_df["ons"] >= 1).sum()
percentage_boarding_4062 = (boarding_stops_4062 / num_stops_4062) * 100 if num_stops_4062 > 0 else 0
print(f"Percentage of stops with boardings on vehicle 4062: {percentage_boarding_4062:.2f}%")

print("\n--- Section 4: Boarding Bias Detection ---")

# 4. find vehicles with biased boarding data ("ons")
from scipy.stats import binomtest

# 4a. system-wide boarding rate
total_stops = len(stops_df)
total_with_boarding = (stops_df["ons"] >= 1).sum()
system_boarding_rate = total_with_boarding / total_stops

# group by vehicle and calculate stats
grouped = stops_df.groupby("vehicle_number")
bias_results = []

for vehicle, group in grouped:
    num_stops = len(group)
    num_boarding = (group["ons"] >= 1).sum()
    p_value = binomtest(num_boarding,num_stops,system_boarding_rate,alternative='two-sided').pvalue
    bias_results.append((vehicle,p_value))

# filter for p < 0.05
biased_vehicles = [(v,p) for v,p in bias_results if p < 0.05]

print("Biased vehicles (p < 0.05):")
for vehicle, p in biased_vehicles:
    print(f"Vehicle {vehicle}, p-value: {p:5f}")

print("\n--- Section 5: GPS Bias Detection ---")

# 5. find vehicles with biased GPS data
import numpy as np
from scipy.stats import ttest_1samp

# load RELPOS data
gps_df = pd.read_csv("trimet_relpos_2022-12-07.csv",parse_dates=["TIMESTAMP"])

# all RELPOS values for full system
all_relpos = gps_df["RELPOS"].values

# group by vehicle and perform t-test against system-wide mean = 0
gps_grouped = gps_df.groupby("VEHICLE_NUMBER")
gps_bias_results = []

for vehicle, group in gps_grouped:
    relpos_values = group["RELPOS"].values
    if len(relpos_values) > 1:
        t_stat, p_value = ttest_1samp(relpos_values, 0.0)
        gps_bias_results.append((vehicle,p_value))

# list vehicles with p < 0.05
biased_gps_vehicles = [(v,p) for v, p in gps_bias_results if p < 0.005]

print("Vehicles with biased GPS data (p < 0.005):")
for vehicle, p in biased_gps_vehicles:
    print(f"Vehicle {vehicle}, p-value: {p:.5f}")
