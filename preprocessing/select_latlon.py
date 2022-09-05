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

lon_min_sel = 13
lon_max_sel = 14
lat_min_sel = 45.5
lat_max_sel = 46.75

i_start = int((lat_min_sel - LAT_MIN) / INTERVAL)
i_end = int((lat_max_sel - LAT_MIN) / INTERVAL)
i_list = np.arange(i_start, i_end, 1)

j_start = int((lon_min_sel - LON_MIN) / INTERVAL)
j_end = int((lon_max_sel - LON_MIN) / INTERVAL)
j_list = np.arange(j_start, j_end, 1)

idx_space_sel = np.array([[i * len(lon_era5_list) + j for j in j_list] for i in i_list])

with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_BIS/gnn_target_2015-2016.pkl', 'rb') as f:
    target = pickle.load(f)

print(f"\nLat range = {lat_min_sel}-{lat_max_sel}, Lon range = {lon_min_sel}-{lon_max_sel}. Starting the processing.")

print(f"\nLen of target = {len(target.keys())}")
for i in range(len(lat_era5_list)):
    for j in range(len(lon_era5_list)):
        idx_space = i * len(lon_era5_list) + j
        if idx_space not in idx_space_sel:
            for t in range(TIME_DIM):
                k = t * 2107 + i * len(lon_era5_list) + j
                if k in target.keys():
                    del target[k]
    print(f"\nLatidute {lat_era5_list[i]} done.")

print(f"\nLen of reduced target = {len(target.keys())}. Starting to write the file.")

with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_BIS/gnn_target_fvg.pkl', 'wb') as f:
    pickle.dump(target, f)

idx_to_key = np.sort(np.array(list(target.keys())))
with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_BIS/idx_to_key_fvg.pkl', 'wb') as f:
    pickle.dump(idx_to_key, f)
