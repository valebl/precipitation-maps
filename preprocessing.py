#--------------------------------
# Import the relevant packages
#--------------------------------

import h5py
import numpy as np
import xarray as xr

#------------
# Functions
#------------

def preprocessing(path, num_partitions, period_of_influence, lat_dim, lon_dim, n_levels, n_variables, idx_normal_to_idx_rand):

    output = np.zeros((num_partitions-1, period_of_influence*2, lat_dim, lon_dim, n_levels*n_variables))
    l_start = 0
    for v in ['q', 't', 'u', 'v', 'z']:
        with xr.open_dataset(path + f'{v}_sliced.nc') as file:
            data = file[v].values
            s = data.shape
            data = data.reshape(s[0], s[2], s[3], s[1])         # (t_dim, lat_dim, lon_dim, n_levels)
            data_split = np.array_split(data, num_partitions)   # list
            idx = 1
            for idx_rand in idx_normal_to_idx_rand: # len(idx_normal_to_idx_rand) num_partitions-1
                output[idx_rand,:,:,:,l_start:l_start+5] = np.concatenate((data_split[idx-1], data_split[idx]),axis=0)
        lstart += 5
        with open('/m100_work/ICT22_ESP_0/vblasone/SLICED/log.txt', 'a') as f:
            f.write(f'\nFinished preprocessing of {v}.')
    
    with open('/m100_work/ICT22_ESP_0/vblasone/SLICED/log.txt', 'a') as f:
        f.write(f'\nStarting to write the output file.')

    with h5py.File(path+'output.hdf5', 'w') as f:
        f.create_dataset('output', output.shape, data=output)

    with open('/m100_work/ICT22_ESP_0/vblasone/SLICED/log.txt', 'a') as f:
        f.write(f'\nPreprocessing finished.')


if __name__ == '__main__':

    path = '/m100_work/ICT22_ESP_0/vblasone/SLICED/'
    year_start = 2001
    year_end = 2016
    period_of_influence = 24 # hours
    n_levels = 5
    n_variables = 5

    with xr.open_dataset(f"{path}/q_sliced.nc") as f:
        hours_total = len(f.time)
        lat_dim = len(f.latitude)
        lon_dim = len(f.longitude)

    num_partitions = int(hours_total / period_of_influence)

    #-----------------------------------------------------------------------------
    # Create an array of random numbers from to (num_partitions - 1)
    #-----------------------------------------------------------------------------

    idx = range(1,num_partitions)
    idx_normal_to_idx_rand = np.random.permutation(idx)             # idx_normal_to_idx_rand[i] = idx of cell i in random array
    idx_rand_to_idx_normal = np.argsort(idx_normal_to_idx_rand)     # idx_rand_to_idx_normal[j] = idx of cell j in normal array

    np.savetxt(path + 'idx_normal_to_idx_rand.csv', idx_normal_to_idx_rand, delimiter=',') # to be able to replicate it
    np.savetxt('idx_rand_to_idx_normal.csv', idx_rand_to_idx_normal, delimiter=',')

    with open('/m100_work/ICT22_ESP_0/vblasone/SLICED/log.txt', 'w') as f:
        f.write(f'\nStarting the preprocessing.')

    preprocessing(path, num_partitions, period_of_influence, lat_dim, lon_dim, n_levels, n_variables, idx_normal_to_idx_rand)

