import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typhon.physics import density
from typhon.plots import worldmap


years = ['2009', '2010', '2021']
months = [f'{month:02d}' for month in range(1, 13)]
for year in years:
    for month in months:
        eml_ds_paths = glob.glob(
                        f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'
                        'monthly_means/geographical/'
                        f'era5_3h_30N-S_eml_tropics_*_{year}-{month}_geographical_mean.nc'
                        )
        eml_ds_paths = np.sort(np.array(eml_ds_paths).flatten())
        for file in eml_ds_paths:
            atlantic_file_name = file[:-3]+'_atlantic.nc'
            print(f"Storing {atlantic_file_name}")
            data = xr.open_dataset(file)
            atlantic_data = data.sel({'lon': slice(-60, 0)})
            atlantic_data.to_netcdf(atlantic_file_name)
