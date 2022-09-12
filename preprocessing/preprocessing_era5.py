import numpy as np
import xarray as xr
import pickle


if __name__ == '__main__':

    input_path = '/m100_work/ICT22_ESP_0/vblasone/SLICED/'
    output_path = '/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/'
    log_file = '/m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/prep_input.txt'
    
    N_LEVELS = 5
    N_VARS = 5
    PAD = 2
    TIME_DIM = 140256
    #LON_MIN = 6.5
    #LON_MAX = 18.75
    #LAT_MIN = 36.5
    #LAT_MAX =  47.25
    #INTERVAL = 0.25

    #LON_LIST = np.arange(LON_MIN, LON_MAX, INTERVAL)
    #LAT_LIST = np.arange(LAT_MIN, LAT_MAX, INTERVAL)
    #SPACE_IDXS_DIM = len(LAT_LIST) * len(LON_LIST)
    #LAT_DIM = len(LAT_LIST)
    #LON_DIM = len(LON_LIST)

    with xr.open_dataset(f"{input_path}/q_sliced.nc") as f:
        lat_dim_era5 = len(f.latitude)
        lon_dim_era5 = len(f.longitude)
    
    with open(output_path+'idx_to_key.pkl', 'rb') as f:
        idx_to_key = pickle.load(f)

    with open(log_file, 'w') as f:
        f.write(f'\nStarting the preprocessing.')

    input_ds = np.zeros((N_VARS, N_LEVELS, TIME_DIM, lat_dim_era5, lon_dim_era5), dtype=np.float32) # variables, levels, time, lat, lon
    v_idx = 0
    for v in ['q', 't', 'u', 'v', 'z']:
        with open(log_file, 'a') as f:
            f.write(f'\nPreprocessing {v}_sliced.nc.')
        with xr.open_dataset(input_path + f'{v}_sliced.nc') as file:
            data = file[v].values
        s = data.shape # (time, levels, lat, lon)
        data = data.reshape(s[1], s[0], s[2], s[3]) # (levels, time, lat, lon)
        input_ds[v_idx,:,:,:,:] = data
        v_idx += 1

    with open(log_file, 'a') as f:
        f.write(f'\nNormalising the dataset.')

    # normalize the dataset
    mean_vars = [np.mean(input_ds[i,:,:,:,:]) for i in range(5)]
    std_vars = [np.std(input_ds[i,:,:,:,:]) for i in range(5)]
    input_ds_standard = np.array([(input_ds[i,:,:,:,:]-mean_vars[i])/std_vars[i] for i in range(5)])

    with open(log_file, 'a') as f:
        f.write(f'\nReshaping the dataset.')

    # reshape the two datasets <-> flatten the levels into channels
    assert input_ds.shape == input_ds_standard.shape
    ds_shape = input_ds.shape
    input_ds = input_ds.reshape(ds_shape[0]*ds_shape[1], ds_shape[2], ds_shape[3], ds_shape[4])
    input_ds_standard = input_ds_standard.reshape(ds_shape[0]*ds_shape[1], ds_shape[2], ds_shape[3], ds_shape[4])

    with open(log_file, 'a') as f:
        f.write(f'\nStarting to write the output file.')
    
    with open(output_path+'input_standard.pkl', 'wb') as f:
      pickle.dump(input_ds_standard, f)
    
    with open(log_file, 'a') as f:
        f.write(f'\n\Output file written.')
        f.write(f'\nPreprocessing finished.')

