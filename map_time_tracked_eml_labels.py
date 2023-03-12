import eval_eml_chars as eec
import moist_layers as ml
import intake
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cartopy import crs as ccrs  # Cartogrsaphy library
import random
import verde as vd

def main():
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask().sel(
        time=slice("2021-07-28T00:00:00", "2021-08-02T00:00:00"))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_eurec4a(ds3d, grid)
    grid = ml.mask_eurec4a(grid, grid)
    times = ds3d.time.values
    region = (-65, -40, 5, 25)
    spacing = 0.05
    grid = vd.grid_coordinates(region, spacing=spacing)
    eml_labels_3d = eec.get_3d_eml_labels(times, grid, spacing)
    random.seed(42)
    colors = [eec.random_color() for i in range(eml_labels_3d.max())]
    colors[0] = (1, 1, 1)
    cmap = ListedColormap(colors, name='eml_label_colors')

    for i, time in enumerate(times):
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
        fig, ax1 = eec.plot_eml_labels(eml_labels_3d[:, :, i], grid[0], grid[1], cmap,
                                   max_eml_label=eml_labels_3d.max(), 
                                   fig=fig, ax=ax1)
        plt.suptitle(f'{str(time)[:19]}')
        plt.savefig(f'plots/time_tracked_eml_labels_{str(time)[:19]}.png') 
    pass

if __name__ == '__main__':
    main()