import os
import numpy as np
import xarray as xr

stat_type = 'all'
year = '2009'
path = f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'\
       f'era5_3h_30N-S_eml_tropics_{year}-01-01T00.nc'
eml_ds = xr.open_dataset(path)

variables = [var for var in list(eml_ds.variables)
             if var not in ['time', 'lat', 'lon', 'fulllevel']] \
                    + ['n_eml'] + ['all']
# variables = ['all']
for month in ['%02d' % i for i in range(1, 13)]:
       for var in variables:
            for region in ['global', 'atlantic', 'west_pacific', 'east_pacific']:
                if var == 'all' and region != 'global':
                      continue
                os.system("sbatch "
                    "/home/u/u300676/moist-layers/batch_moist_layer_calc.sh "
                    f"{year}-{month} {var} {region} {stat_type}")
