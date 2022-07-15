#--------------------------------
# Import the relevant packages
#--------------------------------

import h5py
import numpy as np
import xarray as xr

#------------
# Functions
#------------

def preprocessing_input(input_path, output_path, log_path, lat_dim, lon_dim, n_levels, time_dim, n_variables, year_start, year_end):

    time_start = 0
    for year in range(year_start, year_end +1):
        with xr.open_dataset(input_path + f'q_{year}.nc') as file:
            time_year_dim = len(file.time)
            database = np.zeros((lat_dim, lon_dim, n_levels, time_year_dim, n_variables), dtype=np.float32)
        v_idx = 0
        for v in ['q', 't', 'u', 'v', 'z']:
            with xr.open_dataset(input_path + f'{v}_{year}.nc') as file:
                data = file[v].values
                s = data.shape
                data = data.reshape(s[2], s[3], s[1], s[0])         # (lat_dim, lon_dim, n_levels, time_year_dim)
                database[:,:,:,:,v_idx] = data
                with open(log_path+'log_input.txt', 'a') as f:
                    f.write(f'\nPreprocessing {v}_{year}.nc, time indexes from {time_start} to {time_start+time_year_dim}.')
            v_idx += 1

        with open(log_path+'log_input.txt', 'a') as f:
            f.write(f'\nFinished preprocessing of year {year}.')

        with open(log_path+'log_input.txt', 'a') as f:
            f.write(f'\nStarting to write the output file (year={year}).')
        
        if year == year_start:
            with h5py.File(output_path+'input.hdf5', 'w') as f:
                f.create_dataset('input', database.shape, data=database,
                    compression="gzip",  maxshape=(database.shape[0], database.shape[1], database.shape[2], None, database.shape[4]))
        else:
            with h5py.File(output_path+'input.hdf5','a') as f:
                f['input'].resize((f['input'].shape[3] + database.shape[3]), axis = 3)
                f['input'][:,:,:,-database.shape[3]:,:] = database
        
        with open(log_path+'log_input.txt', 'a') as f:
            f.write(f'\Output file written (year={year}).')
        
        time_start += time_year_dim
        del database

    with open(log_path+'log_input.txt', 'a') as f:
        f.write(f'\nPreprocessing finished.')


if __name__ == '__main__':

    input_path = '/m100_work/ICT22_ESP_0/vblasone/ERA5/'
    output_path = '/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/'
    log_path = '/m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/'
    year_start = 2001
    year_end = 2016
    n_levels = 5
    n_variables = 5
    time_dim = 140256

    with xr.open_dataset(f"{input_path}/q_2001.nc") as f:
        lat_dim = len(f.latitude)
        lon_dim = len(f.longitude)

    with open(log_path+'log_input.txt', 'w') as f:
        f.write(f'\nStarting the preprocessing.')

    preprocessing_input(input_path, output_path, log_path, lat_dim, lon_dim, n_levels, time_dim, n_variables, year_start, year_end)

