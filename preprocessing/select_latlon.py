import numpy as np
import xarray as xr
import pickle
import time
import sys

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

i_start = int(np.where(lat_era5_list == lat_min_sel)[0]) #int((lat_min_sel - LAT_MIN) / INTERVAL)
i_end = int(np.where(lat_era5_list == lat_max_sel)[0]) #int((lat_max_sel - LAT_MIN) / INTERVAL)
i_list = np.arange(i_start, i_end, 1)

j_start = int(np.where(lon_era5_list == lon_min_sel)[0]) #int((lon_min_sel - LON_MIN) / INTERVAL)
j_end = int(np.where(lon_era5_list == lon_max_sel)[0]) #int((lon_max_sel - LON_MIN) / INTERVAL)
j_list = np.arange(j_start, j_end, 1)

idx_space_sel = np.array([[i * len(lon_era5_list) + j for j in j_list] for i in i_list])
#idx_space_sel = idx_space_sel.flatten()

#print(f"\nlat from {lat_era5_list[i_start]} to {lat_era5_list[i_end]}")
#print(f"\nlon from {lon_era5_list[j_start]} to {lon_era5_list[j_end]}")
#print(idx_space_sel.shape)

#sys.exit()

with open('/data/gnn_target.pkl', 'rb') as f:
    target = pickle.load(f)

with open('/data/fvg/log.txt', 'w') as f:
    f.write(f"\nLat range = {lat_era5_list[i_start]}-{lat_era5_list[i_end]}, Lon range = {lon_era5_list[j_start]}-{lon_era5_list[j_end]}. Starting the processing.")

with open('/data/fvg/log.txt', 'a') as f:
    f.write(f"\nLen of target = {len(target.keys())}")

#for i in range(len(lat_era5_list)):
#    for j in range(len(lon_era5_list)):
#        idx_space = i * len(lon_era5_list) + j
#        if idx_space not in idx_space_sel:
#            for t in range(TIME_DIM):
#                k = t * 2107 + idx_space
#                if k in target.keys():
#                    del target[k]
 
target_sel = dict()

for i in i_list:
    for j in j_list:
        idx_space = i * len(lon_era5_list) + j
        if idx_space in idx_space_sel:
            for t in range(TIME_DIM):
                k = t * 2107 + idx_space
                if k in target.keys():
                    target_sel[k] = target[k]

    with open('/data/fvg/log.txt', 'a') as f:
        f.write(f"\nLatidute {lat_era5_list[i]} done.")

with open('/data/fvg/log.txt', 'a') as f:
    f.write(f"\nLen of reduced target = {len(target_sel.keys())}. Starting to write the file.")

with open('/data/fvg/gnn_target_fvg.pkl', 'wb') as f:
    pickle.dump(target_sel, f)

idx_to_key = np.sort(np.array(list(target_sel.keys())))
with open('/data/fvg/idx_to_key_fvg.pkl', 'wb') as f:
    pickle.dump(idx_to_key, f)


