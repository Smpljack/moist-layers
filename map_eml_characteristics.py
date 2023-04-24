import eval_eml_chars as eec
import moist_layers as ml
import intake
import verde as vd
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import argparse
from skimage.measure import regionprops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_start", type=str,
                    help="timestamp",
                    default="2021-08-07T00:00:00")
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
    ds2d = cat["luk1043"].atm2d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_eurec4a(ds3d, grid)
    ds2d = ml.mask_eurec4a(ds2d, grid)
    grid = ml.mask_eurec4a(grid, grid)
    rain_ind = (
        (ds2d.rain_gsp_rate.values != np.nan) & 
        (ds2d.rain_gsp_rate.values > 0.))
    rr_p50 = np.percentile(ds2d.rain_gsp_rate.values[rain_ind]*3600, 50)
    rr_p75 = np.percentile(ds2d.rain_gsp_rate.values[rain_ind]*3600, 75)
    rr_p90 = np.percentile(ds2d.rain_gsp_rate.values[rain_ind]*3600, 90)
    times = ds3d.time.values
    for time in times:
        print(str(time)[:19], flush=True)
        data3d = ds3d.sel(time=time)
        data2d = ds2d.sel(time=time)
        eml_ds = eec.load_eml_data(time)
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=0.3, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=40000,
            )
        region1 = (-65, 0, 0, 25)
        # region2 = (-180, -150, -5, 25)
        spacing = 0.1
        grid1 = vd.grid_coordinates(region1, spacing=spacing)
        # grid2 = vd.grid_coordinates(region2, spacing=spacing)
        # grid = (np.concatenate([grid1[0], grid2[0]], axis=1), 
        #         np.concatenate([grid1[1], grid2[1]], axis=1))
        grid = grid1
        heights = np.arange(2000, 7000, 50)
        eml_labels3d = eec.get_3d_height_eml_labels(eml_ds, grid, spacing,
                                                    heights=heights)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        eml_props = regionprops(eml_labels2d)
        big_emls = [props.label for props in eml_props 
                    if props.area > 500]
        eml_labels2d = np.where(
            np.isin(eml_labels2d, big_emls), eml_labels2d, 0)
        eml_ds_big = eec.get_eml_chars_subset_for_labels(
                eml_ds, eml_labels2d, grid, eml_ds_lat_lon_tuples=None)
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(221, 
                projection=ccrs.PlateCarree())
        gridded_rr = eec.grid_monsoon_data(
            data2d, 'rain_gsp_rate', np.deg2rad(grid[1][:, 0]), np.deg2rad(grid[0][0, :]))
        gridded_q = eec.grid_monsoon_data(
            data3d.sel(fulllevel=68), 'hus', np.deg2rad(grid[1][:, 0]), np.deg2rad(grid[0][0, :]))
        fig, ax1 = eec.plot_gridded_q_map(
            gridded_q, grid[1][:, 0], grid[0][0, :],
            fig=fig, ax=ax1)
        fig, ax1 = eec.plot_rr_contour(
            gridded_rr*3600, grid[1][:, 0], 
            grid[0][0, :], 
            levels=[rr_p50, rr_p75, rr_p90], 
            fig=fig, ax=ax1)
        # ax1.set_extent([-65, 0, 0, 25])
    
        ax2 = fig.add_subplot(
            222, projection=ccrs.PlateCarree(), 
            sharey=ax1)
        fig, ax2 = eec.plot_eml_strength_map(
            eml_ds_big.strength, np.rad2deg(eml_ds_big.lat), 
            np.rad2deg(eml_ds_big.lon), eml_ds_big.time[0].values, fig=fig, ax=ax2)
        fig, ax2 = eec.plot_rr_contour(
            gridded_rr*3600, grid[1][:, 0], 
            grid[0][0, :], 
            levels=[rr_p50, rr_p75, rr_p90], 
            fig=fig, ax=ax2)
        # ax2.set_extent([-65, 0, 0, 25])
        ax3 = fig.add_subplot(
            223, projection=ccrs.PlateCarree(), sharex=ax1)
        fig, ax3 = eec.plot_eml_height_map(
            eml_ds_big.pmean, np.rad2deg(eml_ds_big.lat), 
            np.rad2deg(eml_ds_big.lon), eml_ds_big.time[0].values, fig=fig, ax=ax3)
        fig, ax3 = eec.plot_rr_contour(
            gridded_rr*3600, grid[1][:, 0], 
            grid[0][0, :], 
            levels=[rr_p50, rr_p75, rr_p90], 
            fig=fig, ax=ax3)
        # ax3.set_extent([-65, 0, 0, 25])
        ax4 = fig.add_subplot(
            224, projection=ccrs.PlateCarree(), sharey=ax3)
        fig, ax4 = eec.plot_eml_thickness_map(
            eml_ds_big.pwidth, np.rad2deg(eml_ds_big.lat), 
            np.rad2deg(eml_ds_big.lon), eml_ds_big.time[0].values, fig=fig, ax=ax4)
        fig, ax4 = eec.plot_rr_contour(
            gridded_rr*3600, grid[1][:, 0], 
            grid[0][0, :], 
            levels=[rr_p50, rr_p75, rr_p90], 
            fig=fig, ax=ax4)
        # ax4.set_extent([-65, 0, 0, 25])
        ax1.set_anchor((0, 0.2))
        # plt.subplots_adjust(hspace=-0.5)
        plt.suptitle(f'{str(time)[:19]}')
        # ax1.set_aspect(0.4)
        # ax2.set_aspect(0.4)
        # ax3.set_aspect(0.4)
        # ax4.set_aspect(0.4)
        plt.savefig(f'/home/u/u300676/moist-layers/plots/'
                    f'big_extended_rh_def_composite_eml_map_{str(time)[:19]}.png',
                    dpi=300)
    eec.make_movie(
        image_paths='plots/big_extended_rh_def_composite_eml_map_*.png', 
        video_name='videos/big_extended_rh_def_composite_eml_chars_video.mp4')

if __name__ == '__main__':
    main()
