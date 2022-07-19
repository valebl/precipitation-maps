import numpy as np
import xarray as xr
import pickle

TIME_DIM = 140256
LON_MIN = 6.5
LON_MAX = 18.75
LAT_MIN = 36.5
LAT_MAX =  47.25
INTERVAL = 0.25

def select_from_gripho(lon_min, lon_max, lat_min, lat_max, lon, lat, pr):
    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    selected_pr = np.array([pr[i][bool_both] for i in range(TIME_DIM)])

    return lon[bool_both], lat[bool_both], selected_pr

if __name__ == '__main__':

    gripho = xr.open_dataset('/m100_work/ICT22_ESP_0/vblasone/SLICED/gripho_sliced.nc')
    era5_q = xr.open_dataset('/m100_work/ICT22_ESP_0/vblasone/SLICED/q_sliced.nc')

    target = {}

    # get data from the dataset
    lon = gripho.lon.to_numpy()
    lat = gripho.lat.to_numpy()
    pr = gripho.pr.to_numpy()

    lon_era5_list = np.arange(LON_MIN, LON_MAX, INTERVAL)
    lat_era5_list = np.arange(LAT_MIN, LAT_MAX, INTERVAL)

    i = 0
    for lon_era5 in lon_era5_list:
        j = 0
        for lat_era5 in lat_era5_list:
            selected_lon, selected_lat, selected_pr = select_from_gripho(lon_era5, lon_era5+INTERVAL, lat_era5, lat_era5+INTERVAL, lon, lat, pr)
            target[i, j] = {'longitude' : selected_lon, 'latitude': selected_lat, 'pr': selected_pr, 'lon_era5': lon_era5, 'lat_era5': lat_era5}
            j += 1
        i += 1

with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/target.pkl', 'wb') as f:
    pickle.dump(target, f)
