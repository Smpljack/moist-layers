import eval_eml_chars as eec
import moist_layers as ml
import intake
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cartopy import crs as ccrs  # Cartogrsaphy library
import random
import verde as vd 
from skimage.measure import regionprops, label
import xarray as xr
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_start", type=str,
                    help="timestamp",
                    default="2021-07-28T00:00:00")
    parser.add_argument("--time_end", type=str,
                    help="timestamp",
                    default="2021-08-11T00:00:00")
    args = parser.parse_args()
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_warmpool(ds3d, grid)
    grid = ml.mask_warmpool(grid, grid)
    times = ds3d.time.values
    random.seed(42)
    max_eml_label = 10000
    colors = [eec.random_color() for i in range(max_eml_label)]
    colors[0] = (1, 1, 1)
    cmap = ListedColormap(colors, name='eml_label_colors')
    region1 = (160, 180, 0, 20)
    region2 = (-180, -140, 0, 20)
    spacing = 0.1
    grid1 = vd.grid_coordinates(region1, spacing=spacing)
    grid2 = vd.grid_coordinates(region2, spacing=spacing)
    label_grid = (np.concatenate([grid1[0], grid2[0]], axis=1), 
            np.concatenate([grid1[1], grid2[1]], axis=1))
    # label_grid = grid1
    heights = np.arange(2000, 7000, 100)
    eml_mask2dtime = np.zeros(
        label_grid[0].shape + (len(times),), dtype=np.int64)
    for i, time in enumerate(times):
        print(str(time)[:19], flush=True)
        eml_ds = eec.load_eml_data(time)
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=1e-3, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
        eml_labels3d = eec.get_3d_height_eml_labels(
            eml_ds, label_grid, spacing, heights)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        eml_props = regionprops(eml_labels2d)
        big_emls = [props.label for props in eml_props 
                    if ((props.area_convex > 500))
                    ]
        eml_labels3d = np.where(
            np.isin(eml_labels3d, big_emls), eml_labels3d, 0)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        eml_mask2d = np.where(eml_labels2d != 0, 1, 0) 
        eml_mask2dtime[:, :, i] = eml_mask2d
    eml_labels2dtime = label(eml_mask2dtime)

    for i, time in enumerate(times):
        fig = plt.figure(figsize=(9, 4.5))
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=-170))
        fig, ax1 = eec.plot_eml_labels(
            eml_labels2dtime[:, :, i], label_grid[0], label_grid[1], cmap,
            max_eml_label=max_eml_label, fig=fig, ax=ax1, relabel=False)
        ax1.set_extent([160, -140, 0, 20])
        plt.suptitle(f'{str(time)[:19]}')
        plt.savefig(f'plots/big_warmpool_eml_labels_4d_{str(time)[:19]}.png', 
            dpi=300) 
        
    eec.make_movie(
            'plots/big_warmpool_eml_labels_4d_*.png', 
            'videos/big_warmpool_eml_labels_4d_video.mp4')

if __name__ == '__main__':
    main()
