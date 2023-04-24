import eval_eml_chars as eec
import moist_layers as ml
import intake
import verde as vd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cartopy import crs as ccrs  # Cartogrsaphy library
import random
from skimage.measure import label, regionprops
import xarray as xr
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_start", type=str,
                    help="timestamp",
                    default="2021-07-28T00:00:00")
    parser.add_argument("--time_end", type=str,
                    help="timestamp",
                    default="2021-08-02T00:00:00")
    args = parser.parse_args()
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask().sel(
        time=slice(args.time_start, args.time_start))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_eurec4a(ds3d, grid)
    grid = ml.mask_eurec4a(grid, grid)
    times = ds3d.time.values
    # eml_labels_3d, grid = get_3d_eml_labels(times)
    random.seed(42)
    max_eml_label = 300
    colors = [eec.random_color() for i in range(max_eml_label)]
    colors[0] = (1, 1, 1)
    cmap = ListedColormap(colors, name='eml_label_colors')

    for time in times:
        print(str(time)[:19])
        # data = ds3d.sel(time=time)
        eml_ds = eec.load_eml_data(time)
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=1e-3, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
        region = vd.get_region(
            (np.rad2deg(eml_ds.lon.values), np.rad2deg(eml_ds.lat.values)))
        spacing = 0.05
        grid = vd.grid_coordinates(region, spacing=spacing)
        eml_mask = eec.get_eml_mask(eml_ds, grid=grid, maxdist=spacing*1*111e3)
        eml_labels = label(xr.where(eml_mask, 1, 0), background=0)
        eml_props = regionprops(eml_labels)
        eml_areas = [props.area for props in eml_props] 
        big_emls = [props.label for props in eml_props 
                    if props.area > np.mean(eml_areas)]
        eml_labels = xr.where(np.isin(eml_labels, big_emls), eml_labels, 0)

        fig = plt.figure(figsize=(9, 4.5))
        ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
        fig, ax1 = eec.plot_eml_strength_map(
            eml_ds.strength, np.rad2deg(eml_ds.lat), 
            np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax1)
        ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
        cmap = ListedColormap(colors[:max_eml_label], 
                              name='eml_label_colors')
        fig, ax2 = eec.plot_eml_labels(eml_labels, grid[0], grid[1], cmap,
                                   fig=fig, ax=ax2, relabel=False, 
                                   max_eml_label=max_eml_label)
        plt.suptitle(f'{str(time)[:19]}')
        plt.savefig(f'plots/big_eml_labels_{str(time)[:19]}.png')


if __name__ == '__main__':
    main()