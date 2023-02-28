import os
import numpy as np
import xarray as xr
import intake

cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

ds3d = cat["luk1043"].atm3d.to_dask()
times = ds3d.sel(time=slice("2021-07-28T00:00:00", "2021-08-02T00:00:00")).time.values
for time in times:
    time_str = str(time)[:19]
    os.system("sbatch "
            "/home/u/u300676/moist-layers/batch_moist_layer_calc.sh "
            f"{time_str}")