import glob
import argparse
import numpy as np
import xarray as xr

from typhon.physics import (relative_humidity2vmr,
                            integrate_water_vapor, e_eq_mixed_mk,
                            specific_humidity2vmr)


def add_crh_to_ds(eml_ds, q_var='hus', col_rh_varname='col_rh'):
    eml_ds = eml_ds.assign(
        {col_rh_varname: (('lat', 'lon'), 
        np.full(
            (len(eml_ds.lat), len(eml_ds.lon)), 
            np.nan))}
    )
    
    vmr_sat = xr.DataArray(
        relative_humidity2vmr(
            np.ones(eml_ds.pfull.shape), 
            eml_ds.pfull.values, 
            eml_ds.t.values, 
            e_eq=e_eq_mixed_mk), 
        dims=['fulllevel', 'lat', 'lon']
        )
    vmr_sat_lay = (vmr_sat.where(
            # Free tropospheric humidity
            (eml_ds.pfull < 105000) & (eml_ds.pfull > 5000)).
            dropna(dim='fulllevel', how='all').values[::-1, :, :])
    vmr_lay = (specific_humidity2vmr(eml_ds[f'{q_var}']).where(
            (eml_ds.pfull < 105000) & (eml_ds.pfull > 5000)).
            dropna(dim='fulllevel', how='all').values[::-1, :, :])
    pfull_lay = (eml_ds.pfull.where(
            (eml_ds.pfull < 105000) & (eml_ds.pfull > 5000)).
            dropna(dim='fulllevel', how='all').values[::-1, :, :])
    for ilat, lat in enumerate(eml_ds.lat.values):
        for ilon, lon in enumerate(eml_ds.lon.values):
            vmr_sat_col = vmr_sat_lay[:, ilat, ilon]
            if np.all(np.isnan(vmr_sat_col)):
                print(
                    f'Found no saturation q at lat/lon {lat}/{lon}', 
                    flush=True)
                eml_ds[col_rh_varname][ilat, ilon] = np.nan
                continue
            vmr_sat_col = vmr_sat_col[~np.isnan(vmr_sat_col)]
            vmr_col = vmr_lay[:, ilat, ilon]
            if np.all(np.isnan(vmr_col)):
                print(
                    f'Found no reference q at lat/lon {lat}/{lon}', 
                    flush=True)
                eml_ds[col_rh_varname][ilat, ilon] = np.nan
                continue
            vmr_col = vmr_col[~np.isnan(vmr_col)]
            pfull_col = pfull_lay[:, ilat, ilon]
            if np.all(np.isnan(pfull_col)):
                print(
                    f'Found no pfull at lat/lon {lat}/{lon}', 
                    flush=True)
                eml_ds[col_rh_varname][ilat, ilon] = np.nan
                continue
            pfull_col = pfull_col[~np.isnan(pfull_col)]
            iwv_sat_col = integrate_water_vapor(vmr_sat_col, pfull_col)
            iwv_col = integrate_water_vapor(vmr_col, pfull_col)
            eml_ds[col_rh_varname][ilat, ilon] = iwv_col / iwv_sat_col
    return eml_ds

def add_vwind_to_era5(era5_ds, va):
    era5_ds = era5_ds.assign(
        {
            'va': va
        }
    )
    return era5_ds

def add_rain_rate_to_era5(era5_ds, rain_rate):
    era5_ds = era5_ds.assign(
        {
            'rain_rate': rain_rate
        }
    )
    return era5_ds

def add_era5_var_to_eml_ds(eml_ds, era5_data_array, new_var_name):
    eml_ds = eml_ds.assign(
        {
            new_var_name: era5_data_array
        }
    )
    return eml_ds
    
def add_iwv_to_ds(eml_ds, q_var='hus', iwv_varname='iwv'):
    eml_ds = eml_ds.assign(
        {iwv_varname: (('lat', 'lon'), 
        np.full(
            (len(eml_ds.lat), len(eml_ds.lon)), 
            np.nan))}
    )
    vmr = (specific_humidity2vmr(eml_ds[f'{q_var}']).
            dropna(dim='fulllevel', how='all').values[::-1, :, :])
    pfull = (eml_ds.pfull.
            dropna(dim='fulllevel', how='all').values[::-1, :, :])
    for ilat, lat in enumerate(eml_ds.lat.values):
        for ilon, lon in enumerate(eml_ds.lon.values):
            vmr_col = vmr[:, ilat, ilon]
            if np.all(np.isnan(vmr_col)):
                # print(
                #     f'Found no reference q at lat/lon {lat}/{lon}', 
                #     flush=True)
                eml_ds[iwv_varname][ilat, ilon] = np.nan
                continue
            pfull_col = pfull[:, ilat, ilon]
            no_nan = (~np.isnan(vmr_col) & ~np.isnan(pfull_col))
            vmr_col = vmr_col[no_nan]
            pfull_col = pfull_col[no_nan]
            if np.all(np.isnan(pfull_col)):
                print(
                     'Found no pfull '
                    f'at lat/lon {lat}/{lon}', 
                    flush=True)
                eml_ds[iwv_varname][ilat, ilon] = np.nan
                continue
            iwv_col = integrate_water_vapor(vmr_col, pfull_col) 
            eml_ds[iwv_varname][ilat, ilon] = iwv_col
    return eml_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2021-07")
    args = parser.parse_args()
    base_path = '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
    file_name_head = 'era5_3h_30N-S_eml_tropics_'
    paths = glob.glob(
        base_path + 
        file_name_head + f'{args.time}*[0-9].nc'
        )
    
    for file in paths:
        print(f"Calculating IWV for\n{file}", flush=True)
        eml_ds = xr.open_dataset(file)
        # eml_ds = add_iwv_to_ds(eml_ds, q_var='hus', iwv_varname='iwv') 
        # eml_ds = add_iwv_to_ds(eml_ds, q_var='q_ref', col_rh_varname='ref_iwv') 
        # for var in ['va', 'v']:
        #     if var in list(eml_ds.variables):
        #         eml_ds = eml_ds.drop(var)
        eml_ds = add_crh_to_ds(eml_ds, q_var='q', col_rh_varname='col_rh')
        # eml_ds = add_crh_to_ds(eml_ds, q_var='q_ref', col_rh_varname='ref_col_rh')
        
        print('Storing data!')
        eml_ds.to_netcdf(file[:-3] + '_crh.nc')


if __name__ == '__main__':
    main()