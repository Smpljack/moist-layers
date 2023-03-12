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
    random.seed(42)
    max_eml_label = 1000
    colors = [eec.random_color() for i in range(max_eml_label)]
    colors[0] = (1, 1, 1)
    cmap = ListedColormap(colors, name='eml_label_colors')
    for i, time in enumerate(times):
        print(str(time)[:19])
        eml_ds = eec.load_eml_data(time)
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=1e-4, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
        region = (-65, -40, 5, 25)
        spacing = 0.05
        grid = vd.grid_coordinates(region, spacing=spacing)
        eml_labels3d = eec.get_3d_height_eml_labels(eml_ds, grid, spacing)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        random.seed(42)
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
        fig, ax1 = eec.plot_eml_labels(
            eml_labels2d, grid[0], grid[1], cmap,
            max_eml_label=max_eml_label, fig=fig, ax=ax1, relabel=False)
        plt.suptitle(f'{str(time)[:19]}')
        plt.savefig(f'plots/eml_labels_3d_{str(time)[:19]}.png') 
    pass

if __name__ == '__main__':
    main()
