#--------------------------------
# Import the relevant packages
#--------------------------------

import h5py
import numpy as np
import xarray as xr
import sys

#------------
# Functions
#------------

def preprocessing_input(path, lat_dim, lon_dim, n_levels, time_dim, n_variables, year_start, year_end):

    output = np.zeros((lat_dim, lon_dim, n_levels, time_dim, n_variables)) # '+ 1' is for the target pr value
    # output = np.zeros((num_partitions-1, period_of_influence*2, lat_dim, lon_dim, n_levels*n_variables + 1)) # '+ 1' is for the target pr value
    l_start = 0
    for v in ['q', 't', 'u', 'v', 'z']:
        time_start = 0
        for year in range(year_start, year_end +1):
            with xr.open_dataset(path + f'{v}_{year}.nc') as file:
                time_year_dim = len(file.time)
                data = file[v].values
                s = data.shape
                data = data.reshape(s[2], s[3], s[1], s[0])         # (lat_dim, lon_dim, n_levels, time_year_dim)
                output[:,:,:,time_start:time_start+time_year_dim,l_start:l_start+5] = data
                time_start += time_year_dim
                with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/log_input.txt', 'a') as f:
                    f.write(f'\nPreprocessing {v}_{year}.nc, time indexes from {time_start} to {time_start+time_year_dim}.')
        l_start += 5
        with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/log_input.txt', 'a') as f:
            f.write(f'\nFinished preprocessing of {v}.')
    
    with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/log_input.txt', 'a') as f:
        f.write(f'\nStarting to write the output file.')

    with h5py.File(path+'input.hdf5', 'w') as f:
        f.create_dataset('input', output.shape, data=output)

    with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/log_input.txt', 'a') as f:
        f.write(f'\nPreprocessing finished.')


if __name__ == '__main__':

    path = '/m100_work/ICT22_ESP_0/vblasone/ERA5/'
    year_start = 2001
    year_end = 2016
    n_levels = 5
    n_variables = 5
    time_dim = 140256

    with xr.open_dataset(f"{path}/q_2001.nc") as f:
        lat_dim = len(f.latitude)
        lon_dim = len(f.longitude)

    with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/log_input.txt', 'w') as f:
        f.write(f'\nStarting the preprocessing.')

    preprocessing_input(path, lat_dim, lon_dim, n_levels, time_dim, n_variables, year_start, year_end)

