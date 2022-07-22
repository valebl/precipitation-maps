#--------------------------------
# Import the relevant packages
#--------------------------------

#import h5py
import numpy as np
import xarray as xr
import pickle

#------------
# Functions
#------------

def preprocessing_input(input_path, output_path, log_path, lat_dim, lon_dim, n_levels, time_dim, n_variables, year_start, year_end):

    database = np.zeros((n_variables, n_levels, lon_dim, lat_dim, time_dim, ), dtype=np.float32) # variables, levels, lon, lat, time
    v_idx = 0
    for v in ['q', 't', 'u', 'v', 'z']:
        with xr.open_dataset(input_path + f'{v}_sliced.nc') as file:
            data = file[v].values
            s = data.shape # (time, levels, lat, lon)
            data = data.reshape(s[1], s[3], s[2], s[0])         # (lon_dim, lat_dim, n_levels, time_year_dim)
            database[v_idx,:,:,:,:] = data
            with open(log_path+'log_input.txt', 'a') as f:
                f.write(f'\nPreprocessing {v}_sliced.nc.')
        v_idx += 1

        with open(log_path+'log_input.txt', 'a') as f:
            f.write(f'\nStarting to write the output file.')
    
    #with h5py.File(output_path+'input.hdf5', 'w') as f:
    #        f.create_dataset('input', database.shape, data=database)
    
    with open(output_path+'input.pkl', 'wb') as f:
        pickle.dump(database, f)


    with open(log_path+'log_input.txt', 'a') as f:
        f.write(f'\Output file written.')
    
    del database

    with open(log_path+'log_input.txt', 'a') as f:
        f.write(f'\nPreprocessing finished.')


if __name__ == '__main__':

    input_path = '/m100_work/ICT22_ESP_0/vblasone/SLICED/'
    output_path = '/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/'
    log_path = '/m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/'
    year_start = 2001
    year_end = 2016
    n_levels = 5
    n_variables = 5

    with xr.open_dataset(f"{input_path}/q_sliced.nc") as f:
        time_dim = len(f.time)
        lat_dim = len(f.latitude)
        lon_dim = len(f.longitude)

    with open(log_path+'log_input.txt', 'w') as f:
        f.write(f'\nStarting the preprocessing.')

    preprocessing_input(input_path, output_path, log_path, lat_dim, lon_dim, n_levels, time_dim, n_variables, year_start, year_end)

