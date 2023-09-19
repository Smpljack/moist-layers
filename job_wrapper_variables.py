import os
import numpy as np
import xarray as xr

path = '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'\
       'era5_3h_30N-S_eml_tropics_2021-07-02T00:00:00.nc'
eml_ds = xr.open_dataset(path)

variables = [var for var in list(eml_ds.variables)
             if var not in ['time', 'lat', 'lon', 'fulllevel']] \
                    + ['n_eml']
variables = ['n_eml', 'rh', 'ta', 'pfull', 'ua', 'va']
month = '07'
for var in variables[::-1]:
       for region in ['global']:
              os.system("sbatch "
                     "/home/u/u300676/moist-layers/batch_moist_layer_calc.sh "
                     f"2021-{month} {var} {region} geographical")
