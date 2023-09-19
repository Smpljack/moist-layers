import eval_eml_chars as eec
import moist_layers as ml
import intake
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cartopy import crs as ccrs  # Cartogrsaphy library
import random
import verde as vd 
from skimage.measure import label, regionprops
import numpy as np
import argparse
import xarray as xr

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
    ds2d = cat["luk1043"].atm2d.to_dask().sel(
        time=slice(args.time_start, args.time_start))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_warmpool(ds3d, grid)
    ds2d = ml.mask_warmpool(ds2d, grid)
    grid = ml.mask_warmpool(grid, grid)
    times = ds3d.time.values
    random.seed(42)
    max_eml_label = 10000
    colors = [eec.random_color() for i in range(max_eml_label)]
    colors[0] = (1, 1, 1)
    cmap = ListedColormap(colors, name='eml_label_colors')
    rain_ind = (
        (ds2d.rain_gsp_rate.values != np.nan) & 
        (ds2d.rain_gsp_rate.values > 0.))
    rr_p50 = np.percentile(ds2d.rain_gsp_rate.values[rain_ind]*3600, 50)
    rr_p75 = np.percentile(ds2d.rain_gsp_rate.values[rain_ind]*3600, 75)
    rr_p90 = np.percentile(ds2d.rain_gsp_rate.values[rain_ind]*3600, 90)
    for i, time in enumerate(times):
        print(str(time)[:19], flush=True)
        ds3d_time = ds3d.sel(time=time)
        ds2d_time = ds2d.sel(time=time)
        eml_ds = eec.load_eml_data(time, 'warmpool_rh_def')
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=0.3, 
            min_pmean=20000, 
            max_pmean=40000, 
            min_pwidth=5000, 
            max_pwidth=40000,
            )
        region1 = (150, 180, -5, 25)
        region2 = (-180, -150, -5, 25)
        spacing = 0.1
        grid1 = vd.grid_coordinates(region1, spacing=spacing)
        grid2 = vd.grid_coordinates(region2, spacing=spacing)
        grid = (np.concatenate([grid1[0], grid2[0]], axis=1), 
                np.concatenate([grid1[1], grid2[1]], axis=1))
        # grid = grid1
        heights = np.arange(5000, 14000, 50)
        eml_labels3d = eec.get_3d_height_eml_labels(eml_ds, grid, spacing, 
                                                    heights=heights)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        eml_props = regionprops(eml_labels2d)
        big_emls = [props.label for props in eml_props 
                    if props.area > 500]
        eml_labels2d = np.where(
            np.isin(eml_labels2d, big_emls), eml_labels2d, 0)
        eml_label_profiles = eec.get_mean_profiles_for_labelled_emls(eml_labels2d, grid, ds3d_time)
        eml_label_ds = eec.get_eml_char_ds_for_labelled_emls(
            eml_labels2d, grid, eml_ds)
        fig = plt.figure(figsize=(9, 4.5))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, :2], projection=ccrs.PlateCarree(central_longitude=180))
        fig, ax1 = eec.plot_eml_labels(
            eml_labels2d, grid[0], grid[1], cmap,
            max_eml_label=max_eml_label, fig=fig, ax=ax1, relabel=True)
        gridded_rr = eec.grid_monsoon_data(
            ds2d_time, 'rain_gsp_rate', np.deg2rad(grid[1][:, 0]), np.deg2rad(grid[0][0, :]))
        fig, ax1 = eec.plot_rr_contour(
            gridded_rr*3600, grid[1][:, 0], grid[0][0, :], 
            levels=[rr_p50, rr_p75, rr_p90], 
            fig=fig, ax=ax1)
        ax2 = fig.add_subplot(gs[0, 2])
        for i, label in enumerate(eml_label_profiles.eml_label.values):
            ax2.plot(
                eml_label_profiles.rh_mean[i, :]*100, 
                eml_label_profiles.pfull_mean[i, :]/100,
                color=colors[label])
        ax2.set(ylim=[1020, 50], xlim=[0, 100], 
                ylabel='Pressure / hPa', xlabel='relative humidity / %')
        plt.suptitle(f'{str(time)[:19]}')
        plt.subplots_adjust(wspace=0.5)
        plt.savefig(f'plots/profiles_warmpool_200-500hPa_eml_labels_3d_{str(time)[:19]}.png', 
            dpi=300)
        eec.make_movie(
            'plots/profiles_warmpool_200-500hPa_eml_labels_3d_*.png', 
            'videos/profiles_warmpool_200-500hPa_eml_labels_3d_video.mp4')

if __name__ == '__main__':
    main()
