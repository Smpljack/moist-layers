import xarray as xr
import numpy as np
import glob

years = ['2023']
months = [f'{month:02d}' for month in range(1, 13)]
for year in years:
    for month in months:
        eml_ds_paths = glob.glob(
                        f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'
                        'monthly_means/geographical/'
                        f'era5_3h_30N-S_eml_tropics_pfull_{year}-{month}_geographical_mean.nc'
                        )
        eml_ds_paths = np.sort(np.array(eml_ds_paths).flatten())
        for file in eml_ds_paths:
            file_mean_path = (
                f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'
                 'monthly_means/geographical/'
                f'era5_3h_30N-S_eml_tropics_pfull_mean_{year}-{month}_geographical_mean.nc')
            pfull_mean_data = xr.open_dataset(file).pfull_mean
            print(f"Storing {file_mean_path}")
            pfull_mean_data.to_netcdf(file_mean_path)
