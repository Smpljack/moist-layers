import intake
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import argparse

from typhon.physics import vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk
from typhon.plots import profile_p

import moist_layers as ml


def mask_eurec4a(ds, grid):
    """
    Return EUREC4A domain.
    """
    mask_eurec4a = (
        (grid.clon > np.deg2rad(-65)) &
        (grid.clon < np.deg2rad(-40)) &
        (grid.clat > np.deg2rad(5)) &
        (grid.clat < np.deg2rad(25))
    )
    return ds.isel(cell=mask_eurec4a)

def plot_vmr_profile(p, vmr, vmr_ref=None):
    fig, ax = plt.subplots()
    profile_p(p, vmr, ax=ax, label='vmr profile')
    if vmr_ref is not None:
        profile_p(p, vmr_ref, ax=ax, label='reference')
    return fig, ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_min", type=int,
                    help="cell minimum",
                    default=0)
    parser.add_argument("--cell_max", type=int,
                    help="cell minimum",
                    default=0)
    args = parser.parse_args()
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask()
    # ds2d = cat["luk1043"].atm2d.to_dask()
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = mask_eurec4a(ds3d, grid)
    # ds2d = mask_eurec4a(ds2d, grid)
    grid = mask_eurec4a(grid, grid)
    ds3d = ds3d.sel(time="2021-07-30T06:00:00")
    # ds2d = ds2d.sel(time="2021-07-30T06:00:00")
    # .resample(
    #     time='3h', skipna=True,
    # ).mean()
    eml_chars = []
    for cell in range(args.cell_min, args.cell_max):
        print(cell, flush=True)
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
                    p_min=30000.,
                    p_max=70000.,
                    z_in_km=False,

                    p_in_hPa=False,
                    lat=ds3d_col.clat.values,
                    lon=ds3d_col.clon.values,
                    time=ds3d_col.time.values,
                    )
        eml_chars.append(eml_chars_col)
    ml.save_object_to_pickle(eml_chars, f'eml_chars_cells_{args.cell_min}-{args.cell_max}.pickle')

if __name__ == '__main__':
    main()