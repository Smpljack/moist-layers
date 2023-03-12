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

def grid_monsoon_data(data, grid):
    lon = np.rad2deg(grid.clon.values)
    lat = np.rad2deg(grid.clat.values)
    new_lon = np.arange(-65, -40, 0.05)
    new_lat = np.arange(5, 25, 0.05)
    new_mgrid = np.meshgrid(new_lon, new_lat)
    new_lon_mgrid = new_mgrid[0]
    new_lat_mgrid = new_mgrid[1]
    gridded_ds = xr.Dataset(
        coords=
        {
            'lat': new_lat,
            'lon': new_lon,
            'time': data.time,
            'pfull': data.pfull,
        },
        data_vars=
        {
            f'{var}': (('lon', 'lat'), 
                griddata(
                    (lon, lat), data[f'{var}'], 
                    (new_lon_mgrid, new_lat_mgrid), method='nearest').T)
                for var in data.data_vars
                if var not in ('lat', 'lon', 'time')
        }
    )
    return gridded_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2021-07-30T06:00:00")
    args = parser.parse_args()
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask()
    # ds2d = cat["luk1043"].atm2d.to_dask()
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_eurec4a(ds3d, grid)
    # ds2d = mask_eurec4a(ds2d, grid)
    grid = ml.mask_eurec4a(grid, grid)
    print("Loading data...", flush=True)
    ds3d = ds3d.sel(time=args.time).load()
    # ds2d = ds2d.sel(time="2021-07-30T06:00:00")
    # .resample(
    #     time='3h', skipna=True,
    # ).mean()
    # gridded_ds3d = grid_monsoon_data(ds3d, grid)
    ncells = ds3d.cell.max()
    init_eml_col_flag = True # False until finding first EML case.
    for cell in range(ncells.values):
        print("Calculating moist layer characteristics for cell "
             f"{cell}/{ncells.values} at {args.time}", 
             flush=True)
        ds3d_col = ds3d.sel(
            {
                'cell': cell,
                # 'time': time,
            }
            )
        vmr = specific_humidity2vmr(ds3d_col.hus).values[::-1][:48]
        p = ds3d_col.pfull.values[::-1][:48]
        z = ds3d_col.zg.values[::-1][:48]
        t = ds3d_col.ta.values[::-1][:48]
        vmr_ref = ml.reference_h2o_vmr_profile(vmr, p, z)
        # fig, ax = plot_vmr_profile(p, vmr, vmr_ref)
        # plt.savefig(f'/home/u/u300676/moist-layers/plots/vmr_profile_{cell}.png')
        eml_chars_col = ml.eml_characteristics(vmr, vmr_ref, t, p, z,
                    min_eml_p_width=5000.,
                    min_eml_strength=1e-5,
                    p_min=20000.,
                    p_max=80000.,
                    z_in_km=False,
                    p_in_hPa=False,
                    lat=ds3d_col.clat.values,
                    lon=ds3d_col.clon.values,
                    time=ds3d_col.time.values,
                    )
        # Don't store cases without EMLs (nan=float64 type)
        if type(eml_chars_col.strength[0]) != np.float64:
            if init_eml_col_flag:
                init_eml_col_flag = False
                eml_chars_ds = eml_chars_col.to_xr_dataset()
            else:
                eml_chars_ds = xr.concat(
                    [eml_chars_ds, eml_chars_col.to_xr_dataset()], 
                    dim='eml_count')
    eml_chars_ds.to_netcdf(f'eml_data/eml_chars_extended_{args.time}.nc')

if __name__ == '__main__':
    main()