import numpy as np
import xarray as xr
import pickle
import time
import sys

input_path = "/data/" # change this is needed
output_path = "/data/north/" # change this if needed

TIME_DIM = 140256
LON_MIN = 6.5
LON_MAX = 18.75
LAT_MIN = 36.5
LAT_MAX = 47.25
INTERVAL = 0.25

lon_era5_list = np.arange(LON_MIN, LON_MAX, INTERVAL)
lat_era5_list = np.arange(LAT_MIN, LAT_MAX, INTERVAL)

### FVG
#lon_min_sel = 13
#lon_max_sel = 14
#lat_min_sel = 45.5
#lat_max_sel = 46.75

### TRIVENETO
#lon_min_sel = 10.25
#lon_max_sel = 14
#lat_min_sel = 44.75
#lat_max_sel = 47.25

### NORTH-ITALY
lon_min_sel = 6.5
lon_max_sel = 14.00
lat_min_sel = 43.75
lat_max_sel = 47.25

i_start = int((lat_min_sel - LAT_MIN) / INTERVAL) #int(np.where(lat_era5_list == lat_min_sel)[0]) 
i_end = int((lat_max_sel - LAT_MIN) / INTERVAL) #int(np.where(lat_era5_list == lat_max_sel)[0])
i_list = np.arange(i_start, i_end, 1)

j_start = int((lon_min_sel - LON_MIN) / INTERVAL) # int(np.where(lon_era5_list == lon_min_sel)[0]) 
j_end = int((lon_max_sel - LON_MIN) / INTERVAL) # int(np.where(lon_era5_list == lon_max_sel)[0])
j_list = np.arange(j_start, j_end, 1)

idx_space_sel = np.array([[i * len(lon_era5_list) + j for j in j_list] for i in i_list])
idx_space_sel = idx_space_sel.flatten()

with open(input_path + 'gnn_target.pkl', 'rb') as f:
    target = pickle.load(f)

with open(output_path + 'log.txt', 'w') as f:
    f.write(f"\nNumber of lon points: {len(lon_era5_list)}. Number of lat points: {len(lat_era5_list)}.\nLat range = {lat_era5_list[i_start]}-{lat_era5_list[i_end-1]+0.25}, Lon range = {lon_era5_list[j_start]}-{lon_era5_list[j_end-1]+0.25}. Starting the processing.")

with open(output_path + 'log.txt', 'a') as f:
    f.write(f"\nLen of target = {len(target.keys())}")

target_sel = dict()

for i in i_list:
    for j in j_list:
        idx_space = i * len(lon_era5_list) + j
        if idx_space in idx_space_sel:
            for t in range(TIME_DIM):
                k = t * 2107 + idx_space
                if k in target.keys():
                    target_sel[k] = target[k]

    with open(output_path + 'log.txt', 'a') as f:
        f.write(f"\nLatidute {lat_era5_list[i]} done.")

with open(output_path + 'log.txt', 'a') as f:
    f.write(f"\nLen of reduced target = {len(target_sel.keys())}. Starting to write the file.")

with open(output_path + 'gnn_target_north.pkl', 'wb') as f:
    pickle.dump(target_sel, f)

idx_to_key = np.sort(np.array(list(target_sel.keys())))
with open(output_path + 'idx_to_key_north.pkl', 'wb') as f:
    pickle.dump(idx_to_key, f)


