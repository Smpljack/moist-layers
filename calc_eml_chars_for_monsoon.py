import intake
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import griddata

from typhon.physics import (vmr2relative_humidity, specific_humidity2vmr, 
                            e_eq_mixed_mk, vmr2specific_humidity)
from typhon.plots import profile_p

import moist_layers as ml
import eval_eml_chars as eec

def plot_vmr_profile(p, vmr, vmr_ref=None):
    fig, ax = plt.subplots()
    profile_p(p, vmr, ax=ax, label='vmr profile')
    if vmr_ref is not None:
        profile_p(p, vmr_ref, ax=ax, label='reference')
    return fig, ax

def grid_2d(ds2d, var, grid, new_lon, new_lat):
    lon = np.rad2deg(grid.clon.values)
    lat = np.rad2deg(grid.clat.values)
    new_mgrid = np.meshgrid(new_lon, new_lat)
    new_lon_mgrid = new_mgrid[0]
    new_lat_mgrid = new_mgrid[1]
    gridded_ds = xr.Dataset(
        coords=
        {
            'lat': (('lat'), new_lat),
            'lon': (('lon'), new_lon),
            'time': (('time'), np.array([ds2d.time.data])),
        }, 
        data_vars=
        {
            f'{var}': (('lat', 'lon'), 
                griddata(
                    (lon, lat), ds2d[f'{var}'].T.data, 
                    (new_lon_mgrid, new_lat_mgrid), method='linear'))}
                    )
    return gridded_ds

def grid_monsoon_data(data, grid, new_lon, new_lat):
    lon = np.rad2deg(grid.clon.values)
    lat = np.rad2deg(grid.clat.values)
    new_mgrid = np.meshgrid(new_lon, new_lat)
    new_lon_mgrid = new_mgrid[0]
    new_lat_mgrid = new_mgrid[1]
    gridded_ds = xr.Dataset(
        coords=
        {
            'lat': (('lat'), new_lat),
            'lon': (('lon'), new_lon),
            'time': (('time'), np.array([data.time.data])),
            'fulllevel': (('fulllevel'), data.fulllevel.data),
        },
        data_vars=
        {
            f'{var}': (('lat', 'lon', 'fulllevel'), 
                griddata(
                    (lon, lat), data[f'{var}'].T.data, 
                    (new_lon_mgrid, new_lat_mgrid), method='linear'))
                for var in data.data_vars
                if var not in 
                ('lat', 'lon', 'time', 'wa', 'zghalf', 'unknown')
        }
    )
    gridded_ds = gridded_ds.assign(
        {
            'wa': (('lat', 'lon', 'halflevel'), 
                griddata(
                    (lon, lat), data[f'wa'].T.data, 
                    (new_lon_mgrid, new_lat_mgrid), method='linear')),
            'zhalf': (('lat', 'lon', 'halflevel'), 
                griddata(
                    (lon, lat), data[f'zghalf'].T.data, 
                    (new_lon_mgrid, new_lat_mgrid), method='linear')) 
                    
        }
    )
    return gridded_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2021-07-28T00:00:00")
    args = parser.parse_args()
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask()
    ds2d = cat["luk1043"].atm2d.to_dask()
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    new_lon = np.arange(-180, 180., 0.25)
    new_lat = np.arange(-30, 30.25, 0.25)
    ds3d = ml.mask_tropics(ds3d, grid).drop(
        ['unknown'])
    ds2d = ml.mask_tropics(ds2d, grid)
    grid = ml.mask_tropics(grid, grid)
    print("Loading data...", flush=True)
    ds3d = ds3d.sel(time=args.time).load()

    ds2d = ds2d.resample(
        time='3h', skipna=True).mean().sel(time=args.time)
    print("Gridding data...", flush=True)
    gridded_ds3d = grid_monsoon_data(ds3d, grid, new_lon, new_lat)
    gridded_rr = grid_2d(ds2d, 'rain_gsp_rate', grid, new_lon, new_lat)
    gridded_ds3d = gridded_ds3d.assign(
        {'vmr_h2o': (('lat', 'lon', 'fulllevel'), 
                     specific_humidity2vmr(gridded_ds3d.hus).values),
         'rh': (('lat', 'lon', 'fulllevel'), 
                 vmr2relative_humidity(
                    specific_humidity2vmr(gridded_ds3d.hus).values, 
                    gridded_ds3d.pfull.values, 
                    gridded_ds3d.ta.values, 
                    e_eq=e_eq_mixed_mk) ),
        'rain_gsp_rate': gridded_rr.rain_gsp_rate
        })
    gridded_ds3d = gridded_ds3d.assign(
        {'rain_gsp_rate':gridded_rr.rain_gsp_rate}) 
    gridded_ds3d = gridded_ds3d.assign(
        {
            f'{eml_var}': (('lat', 'lon', 'fulllevel'), 
                            np.full(gridded_ds3d.hus.shape, np.nan))
            for eml_var in 
            ['eml_strength', 'eml_pmean', 'eml_pwidth', 
             'eml_pmax', 'eml_pmin', 'eml_zmean', 'eml_zmax', 'eml_zmin',
             'q_ref']
        }
    )
    for ilat, lat in enumerate(gridded_ds3d.lat):
        print("Calculating moist layer characteristics for lat"
             f"{np.round(lat.values, 2)} at {args.time}", 
             flush=True)
        for ilon, lon in enumerate(gridded_ds3d.lon):
            ds3d_col = gridded_ds3d.sel(
                {
                    'lat': lat,
                    'lon': lon,
                    # 'time': time,
                }
                ).squeeze()
            vmr = specific_humidity2vmr(ds3d_col.hus).values[::-1][:48]
            p = ds3d_col.pfull.values[::-1][:48]
            z = ds3d_col.zg.values[::-1][:48]
            if np.all(np.isnan(z)):
                continue
            t = ds3d_col.ta.values[::-1][:48]
            vmr_ref = ml.reference_h2o_vmr_profile(vmr, p, z, 
                            from_mixed_layer_top=True)
            gridded_ds3d['q_ref'][ilat, ilon, 42:] = \
                    vmr2specific_humidity(vmr_ref)[::-1]
            # fig, ax = plot_vmr_profile(p, vmr, vmr_ref)
            # plt.savefig(f'/home/u/u300676/moist-layers/plots/vmr_profile_{cell}.png')
            eml_chars_col = ml.eml_characteristics(vmr, vmr_ref, t, p, z,
                        min_eml_p_width=5000.,
                        min_eml_strength=0.2,
                        p_min=20000.,
                        p_max=90000.,
                        z_in_km=False,
                        p_in_hPa=False,
                        lat=ds3d_col.lat.values,
                        lon=ds3d_col.lon.values,
                        time=ds3d_col.time.values,
                        )
            
            if np.isnan(eml_chars_col.strength[0]):
                continue
            # vertical gridding of emls
            gridded_eml_ds = eec.eml_char_ds_to_pgrid(
                    eml_chars_col.to_xr_dataset(), 
                    ds3d_col.pfull.values,
                    vertical_dim='fulllevel') 
            for eml_var in ['strength', 'pwidth', 'pmax', 'pmin', 
                            'zmean', 'zmax', 'zmin']:
                gridded_ds3d[f'eml_{eml_var}'][ilat, ilon, :] = \
                    gridded_eml_ds[f'{eml_var}'].values
                    # Don't store cases without EMLs (nan=float64 type) 
    gridded_ds3d = gridded_ds3d.transpose(
        'time', 'fulllevel', 'halflevel', 'lat', 'lon', missing_dims='warn')
    gridded_ds3d.to_netcdf('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                           f'gridded_monsoon_0p25deg_eml_tropics_{args.time}.nc')

if __name__ == '__main__':
    main()