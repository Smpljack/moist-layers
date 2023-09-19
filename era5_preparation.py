import xarray as xr

import eval_eml_chars as eec
from calc_eml_chars_for_era5 import prepare_era5_ds

def main():
    era5_paths_lnsp = [
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-07_152_monmean_remapped.nc',
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-08_152_monmean_remapped.nc']
    era5_paths_q = [
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-07_133_monmean_remapped.nc',
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-08_133_monmean_remapped.nc']
    era5_paths_t = [
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-07_130_monmean_remapped.nc',
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-08_130_monmean_remapped.nc']
    era5_paths_v = [
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-07_132_monmean_remapped.nc',
    '/home/u/u300676/user_data/mprange/era5_tropics/' 
    'era5_tropics_2021-08_132_monmean_remapped.nc'] 

    era5_lnsp = xr.open_mfdataset(
        era5_paths_lnsp, combine='by_coords'
        ).lnsp.resample(time='1M').mean()
    era5_q = xr.open_mfdataset(
        era5_paths_q, combine='by_coords'
        ).q.resample(time='1M').mean()
    era5_t = xr.open_mfdataset(
        era5_paths_t, combine='by_coords'
        ).t.resample(time='1M').mean()
    era5_v = xr.open_mfdataset(
        era5_paths_v, combine='by_coords'
        ).v.resample(time='1M').mean()
    
    for month in [7, 8]:
        lnsp = era5_lnsp.sel(
            time='2021-'+'%02d'%month).squeeze('time').load()
        q = era5_q.sel(time='2021-'+'%02d'%month).squeeze('time').load()
        t = era5_t.sel(time='2021-'+'%02d'%month).squeeze('time').load()
        v = era5_v.sel(time='2021-'+'%02d'%month).squeeze('time').load()
        era5_ds = prepare_era5_ds(lnsp, q, t, v)
        era5_ds.to_netcdf(
            '/home/u/u300676/user_data/mprange/era5_tropics/' 
            'era5_tropics_2021-'+'%02d'%month+'_mean.nc')

f __name__ == '__main__':
    main()
