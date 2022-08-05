import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import time

def select_from_gripho(lon_min, lon_max, lat_min, lat_max, lon, lat, pr, geo):
    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    selected_pr = np.array([pr[i][bool_both] for i in range(TIME_DIM)])

    return lon[bool_both], lat[bool_both], selected_pr, geo[bool_both]

if __name__ == '__main__':

    log_file = '/m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/prep_gripho_and_topo_external.txt'

    TIME_DIM = 140256
    SPATIAL_POINTS_DIM = 2107
    LON_MIN = 6.5
    LON_MAX = 18.75
    LAT_MIN = 36.5
    LAT_MAX =  47.25
    INTERVAL = 0.25

    LON_DIFF_MAX = 0.25 / 8 * 2
    LAT_DIFF_MAX = 0.25 / 10 * 2


    gripho = xr.open_dataset('/m100_work/ICT22_ESP_0/vblasone/GRIPHO/gripho-v1_1h_TSmin30pct_2001-2016_cut.nc')
    era5_q = xr.open_dataset('/m100_work/ICT22_ESP_0/vblasone/SLICED/q_sliced.nc')
    topo = xr.open_dataset('/m100_work/ICT22_ESP_0/vblasone/TOPO/GMTED_DEM_30s_remapdis_GRIPHO.nc')

    lon = gripho.lon.to_numpy()
    lat = gripho.lat.to_numpy()
    pr = gripho.pr.to_numpy()
    geo = topo.z.to_numpy()

    lon_era5_list = np.arange(LON_MIN, LON_MAX, INTERVAL)
    lat_era5_list = np.arange(LAT_MIN, LAT_MAX, INTERVAL)

    with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/gnn_data.pkl', 'rb') as f:
        data_internal = pickle.load(f)

    with open(log_file, 'w') as f:
        f.write(f'\nStarting the preprocessing.')
    

    gnn_target_external = {}
    gnn_data_external = {}
    start = time.time()
    i = 0
    for lon_era5 in lon_era5_list:
        j = 0
        for lat_era5 in lat_era5_list:
            selected_lon, selected_lat, selected_pr, selected_geo = select_from_gripho(
                lon_era5, lon_era5+INTERVAL, lat_era5-INTERVAL, lat_era5, lon, lat, pr, geo)
            assert selected_lon.shape == selected_geo.shape

            idx_space = i * lat_era5_list.shape[0] + j
            if idx_space in data_internal.keys(): # we only want to consider the cells on the perimeter
                print(f'idx {idx_space} is an internal index')
                continue
            elif selected_pr.size == 0 or np.isnan(selected_pr).all(): # cell must contain a part of Italy
                print('either there are no pr values or they all are nan')
                continue
            else:
                print('there are some nice nodes!')
                selected_pr_italy = []
                selected_lon_italy = []
                selected_lat_italy = []
                selected_geo_italy = []
                for s in range(selected_pr.shape[1]):
                    # for each (lon, lat) point check that there is at least one value not nan
                    if not np.isnan(selected_pr[:,s]).all():
                        selected_pr_italy.append(selected_pr[:,s])
                        selected_lon_italy.append(selected_lon[s])
                        selected_lat_italy.append(selected_lat[s])
                        selected_geo_italy.append(selected_geo[s])

                selected_pr_italy = np.array(selected_pr_italy)
                selected_pr_italy = selected_pr_italy.reshape(
                    selected_pr_italy.shape[1],selected_pr_italy.shape[0])


                x = np.stack((selected_lon_italy, selected_lat_italy, selected_geo_italy), axis=-1)
                edge_index = np.empty((2,0), dtype=int)
                ii = 0
                for xi in x:
                    jj = 0
                    for xj in x:
                        if not np.array_equal(xi, xj) and np.abs(xi[0] - xj[0]) < LON_DIFF_MAX and np.abs(xi[1] - xj[1]) < LAT_DIFF_MAX:
                            edge_index = np.concatenate((edge_index, np.array([[ii], [jj]])), axis=-1, dtype=int)
                        jj += 1
                    ii += 1
                gnn_data_external[(i * lat_era5_list.shape[0] + j)] = {'x': x, 'edge_index': edge_index}

                for t in range(TIME_DIM):
                        idx = t * 2107 + idx_space
                        if not np.isnan(selected_pr_italy[t]).any():
                            pr_t = selected_pr_italy[t]
                            y = pr_t.reshape(pr_t.shape[0],1)
                            gnn_target_external[idx] = y
            j+=1
        with open(log_file, 'a') as f:
            f.write(f"Finished longitude = {lon_era5}.")
        i += 1   

    with open(log_file, 'a') as f:
        f.write(f"\nPreprocessing took {time.time() - start} seconds")    

    with open(log_file, 'a') as f:
        f.write(f'\nStarting to write the file.')    

    with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/gnn_target_external.pkl', 'wb') as f:
        pickle.dump(gnn_target_external, f)
    
    with open('/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/gnn_data_external.pkl', 'wb') as f:
        pickle.dump(gnn_data_external, f)
