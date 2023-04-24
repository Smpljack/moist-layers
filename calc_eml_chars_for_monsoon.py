import intake
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import argparse
from scipy.interpolate import griddata

from typhon.physics import vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk
from typhon.plots import profile_p

import moist_layers as ml

def plot_vmr_profile(p, vmr, vmr_ref=None):
    fig, ax = plt.subplots()
    profile_p(p, vmr, ax=ax, label='vmr profile')
    if vmr_ref is not None:
        profile_p(p, vmr_ref, ax=ax, label='reference')
    return fig, ax

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
            'pfull': (('lat', 'lon', 'fulllevel'), 
            griddata(
                (lon, lat), data['pfull'].T.data, 
                (new_lon_mgrid, new_lat_mgrid), method='nearest')),
        },
        data_vars=
        {
            f'{var}': (('lat', 'lon', 'fulllevel'), 
                griddata(
                    (lon, lat), data[f'{var}'].T.data, 
                    (new_lon_mgrid, new_lat_mgrid), method='nearest'))
                for var in data.data_vars
                if var not in ('lat', 'lon', 'time', 'pfull', 'wa', 'zghalf')
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
    # ds2d = cat["luk1043"].atm2d.to_dask()
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    new_lon = np.arange(-180, 180, 0.1)
    new_lat = np.arange(-20, 20, 0.1)
    ds3d = ml.mask_tropics(ds3d, grid)
    # ds2d = mask_eurec4a(ds2d, grid)
    grid = ml.mask_tropics(grid, grid)
    print("Loading data...", flush=True)
    ds3d = ds3d.sel(time=args.time).load()
    # ds2d = ds2d.sel(time="2021-07-30T06:00:00")
    # .resample(
    #     time='3h', skipna=True,
    # ).mean()
    print("Gridding data...", flush=True)
    gridded_ds3d = grid_monsoon_data(ds3d, grid, new_lon, new_lat)
    gridded_ds3d = gridded_ds3d.assign(
        {
            f'{eml_var}': (('lat', 'lon', 'pfull'), 
                            np.full(gridded_ds3d.hus.shape, np.nan))
            for eml_var in 
            ['eml_strength', 'eml_pmean', 'eml_pwidth', 
             'eml_pmax', 'eml_pmin', 'eml_zmean', 'eml_zmax', 'eml_zmin']
            
        }
    )
    for ilat, lat in enumerate(gridded_ds3d.lat):
        for ilon, lon in enumerate(gridded_ds3d.lon):
            print("Calculating moist layer characteristics for lat/lon "
                f"{np.round(lat.values, 2)}/{np.round(lon.values, 2)} " 
                f"at {args.time}", 
                flush=True)
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
            t = ds3d_col.ta.values[::-1][:48]
            vmr_ref = ml.reference_h2o_vmr_profile(vmr, p, z, 
                            from_mixed_layer_top=True)
            # fig, ax = plot_vmr_profile(p, vmr, vmr_ref)
            # plt.savefig(f'/home/u/u300676/moist-layers/plots/vmr_profile_{cell}.png')
            eml_chars_col = ml.eml_characteristics(vmr, vmr_ref, t, p, z,
                        min_eml_p_width=5000.,
                        min_eml_strength=0.2,
                        p_min=20000.,
                        p_max=80000.,
                        z_in_km=False,
                        p_in_hPa=False,
                        lat=ds3d_col.lat.values,
                        lon=ds3d_col.lon.values,
                        time=ds3d_col.time.values,
                        )
            # vertical gridding of emls
            gridded_eml_ds = (eml_chars_col.
                to_xr_dataset().
                assign_coords({'pmean': eml_chars_col.to_xr_dataset().pmean}).
                swap_dims({'eml_count': 'pmean'}).
                reindex(
                    indexers={'pmean': ds3d_col.pfull.values}, 
                    method='nearest', 
                    tolerance=1000).
                swap_dims({'pmean': 'fulllevel'})
                )
            for eml_var in ['strength', 'pmean', 'pwidth', 'pmax', 'pmin', 
                            'zmean', 'zmax', 'zmin']:
                gridded_ds3d[f'eml_{eml_var}'][ilat, ilon, :] = \
                    gridded_eml_ds[f'{eml_var}']
                    # Don't store cases without EMLs (nan=float64 type) 
    gridded_ds3d.to_netcdf('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                           f'gridded_monsoon_eml_tropics_{args.time}.nc')

if __name__ == '__main__':
    main()