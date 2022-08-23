import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import time

TIME_DIM = 140256
LON_MIN = 6.5
LON_MAX = 18.75
LAT_MIN = 36.5
LAT_MAX =  47.25
INTERVAL = 0.25

lon_era5_list = np.arange(LON_MIN, LON_MAX, INTERVAL)
lat_era5_list = np.arange(LAT_MIN, LAT_MAX, INTERVAL)

YEAR_MAX = 2014
time_max = int(TIME_DIM / 16 * (YEAR_MAX - 2000))

with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_BIS/gnn_target_2015-2016.pkl', 'rb') as f:
    target = pickle.load(f)

print(f"\nTime max = {time_max}. Starting the processing.")

print(f"\nLen of target = {len(target.keys())}")
for i in range(len(lat_era5_list)):
    for j in range(len(lon_era5_list)):
       for t in range(time_max):
            k = t * 2107 + i * len(lon_era5_list) + j
            if k in target.keys():
                del target[k]
    print(f"\nLatidute {lat_era5_list[i]} done.")

print(f"\nLen of reduced target = {len(target.keys())}. Starting to write the file.")

with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_BIS/gnn_target_2015-2016.pkl', 'wb') as f:
    pickle.dump(target, f)

idx_to_key = np.sort(np.array(list(target.keys())))
with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_BIS/idx_to_key_2015-2016.pkl', 'wb') as f:
    pickle.dump(idx_to_key, f)
