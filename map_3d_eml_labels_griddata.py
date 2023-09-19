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
import glob

from typhon.physics import (specific_humidity2vmr, 
                            vmr2relative_humidity, 
                            e_eq_mixed_mk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2021-07-19")
    args = parser.parse_args()
    eml_ds_paths = sorted(
        glob.glob('/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                  'era5_3h_30N-S_eml_tropics_2021-07-19*.nc'))

    eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested').sel(
            time=args.time)
    eml_ds = eml_ds.sel({'lat': slice(30, 0), 'lon': slice(-70, 20)})
    # eml_ds = eml_ds.transpose('time', 'fulllevel', 'lat', 'lon')
    # eml_ds = eml_ds.assign(
    #     {'vmr_h2o': (('time', 'fulllevel', 'lat', 'lon'), 
    #                  specific_humidity2vmr(eml_ds.hus.values)),
    #      'rh': (('time', 'fulllevel', 'lat', 'lon'), 
    #              vmr2relative_humidity(
    #                 specific_humidity2vmr(eml_ds.hus).values, 
    #                 eml_ds.pfull.values, 
    #                 eml_ds.ta.values, 
    #                 e_eq=e_eq_mixed_mk) )
    #     })
    # eml_ds = eml_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    times = eml_ds.time.values
    random.seed(42)
    n_colors = 10000
    colors = [eec.random_color() for i in range(n_colors)]
    colors[0] = (1, 1, 1)  
    for i, time in enumerate(times):
        print(str(time)[:19], flush=True)
        eml_ds_time = eml_ds.sel(time=time)
        eml_ds_time_prange = eml_ds_time.where(
            (eml_ds_time.pfull > 50000) &
            (eml_ds_time.pfull < 70000)
        )
        eml_ds_time = eml_ds_time.where(
            np.any(eml_ds_time_prange.eml_strength > 0.3, axis=0) &
            np.any(eml_ds_time_prange.eml_pwidth < 40000, axis=0)
            )

        heights = np.arange(2000, 6000, 50)
        eml_labels3d = eec.get_3d_height_eml_labels_griddata(
            eml_ds_time, eml_ds_time.lat, eml_ds_time.lon, heights=heights,
            height_tolerance=50)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        eml_props = regionprops(eml_labels2d)
        big_emls = [props.label for props in eml_props 
                    if props.area > 100]
        eml_labels2d = np.where(
            np.isin(eml_labels2d, big_emls), eml_labels2d, 0)
        eml_labels2d = eec.relabel(eml_labels2d)
        n_emls = eml_labels2d.max()
        cmap = ListedColormap(
            colors[:n_emls+1], name='eml_label_colors', N=n_emls)
        eml_label_profiles = \
            eec.get_mean_profiles_for_labelled_emls_griddata(
                eml_labels2d, eml_ds_time, vertical_dim='fulllevel')
        fig = plt.figure(figsize=(21, 7))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
        fig, ax1 = eec.plot_eml_labels(
            eml_labels2d, eml_ds_time.lon, eml_ds_time.lat, cmap,
            max_eml_label=n_emls, fig=fig, ax=ax1)
        ax1.set_ylim([0, 30])
        # ax1.set_aspect(0.)
        plt.savefig(f'plots/era5/era5_500-700hPa_map_strength_0p3_eml_labels_3d_{str(time)[:19]}.png', 
            dpi=300)
        fig2 = plt.figure(figsize=(5, 15)) 
        gs = fig.add_gridspec(1, 1)
        ax2 = fig2.add_subplot(gs[0, 0])
        for i, label in enumerate(eml_label_profiles.eml_label.values):
            ax2.plot(
                eml_label_profiles.rh_mean[i, :]*100, 
                eml_label_profiles.pfull_mean[i, :]/100,
                color=colors[label])
        ax2.set(ylim=[1020, 50], xlim=[0, 100], 
                ylabel='Pressure / hPa', xlabel='relative humidity / %')
        plt.suptitle(f'{str(time)[:19]}')
        # plt.subplots_adjust(wspace=0.5)
        plt.savefig(f'plots/era5/era5_500-700hPa_profiles_strength_0p3_eml_labels_3d_{str(time)[:19]}.png', 
            dpi=300)
        # eec.make_movie(
        #     'plots/era5/500-800hPa_strength_0p3_eml_labels_3d_*.png', 
        #     'videos/monsoon_gridded_500-800hPa_strength_0p3_eml_labels_3d_video.mp4')

if __name__ == '__main__':
    main()
