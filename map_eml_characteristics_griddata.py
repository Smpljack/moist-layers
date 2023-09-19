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
import glob 
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="time",
                    default="2021-07-20")
    args = parser.parse_args()
    eml_ds_paths = sorted(
        glob.glob('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                  'gridded_monsoon_0p25deg_eml_tropics_2021-07-20*.nc'))
    eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested').sel(
            time=args.time
        )
    # eml_ds = eml_ds.sel({'lat': slice(30, 0), 'lon': slice(-70, 20)})
    # eml_ds = eml_ds.rename(
    #     {'lat': 'latitude', 
    #     'lon': 'longitude',
    #     'hus': 'q'}).transpose('time', 'fulllevel', 'latitude', 'longitude')
    times = eml_ds.time.values 
    for time in times:
        print(str(time)[:19], flush=True)
        eml_ds_time = eml_ds.sel(time=time).load()
        eml_ds_time = eec.filter_gridded_eml_ds(
            eml_ds_time, 
            min_strength=0.3, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=5000, 
            max_pwidth=40000) 
        heights = np.arange(2000, 7000, 50)
        eml_labels3d = eec.get_3d_height_eml_labels_griddata(
            eml_ds_time, eml_ds_time.lat, eml_ds_time.lon, heights=heights,
            height_tolerance=50)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        eml_props = regionprops(eml_labels2d)
        big_emls = [props.label for props in eml_props 
                    if props.area > 10]
        # eml_labels2d = np.where(
        #     np.isin(eml_labels2d, big_emls), eml_labels2d, 0)
        eml_ds_time_big = eml_ds_time.where(
            xr.DataArray(
                data=eml_labels2d, dims=('lat', 'lon')) != 0)
        fig = plt.figure(figsize=(24, 4))
        # ax1 = fig.add_subplot(411, 
        #         projection=ccrs.PlateCarree()) 
        # q500 = eml_ds_time.q.sel(
        #     lev=np.abs(eml_ds_time.pfull - 50000).argmin(
        #         dim='lev')).values
        # fig, ax1 = eec.plot_gridded_q_map(
        #     q500, eml_ds_time.latitude, eml_ds_time.longitude,
        #     fig=fig, ax=ax1)
        # ax1.set_extent([-65, 0, 0, 25])

        ax2 = fig.add_subplot(
            111, projection=ccrs.PlateCarree(), )
            # sharey=ax1)
        max_strength = eml_ds_time_big.eml_strength.max(dim='fulllevel') 
        max_strength_bool = np.argmax(
            eml_ds_time_big.eml_strength.values == max_strength.values, axis=0)
        max_strength_ind = xr.DataArray(
            np.where(max_strength_bool, max_strength_bool, 1), 
            dims=['lat', 'lon'])
        fig, ax2 = eec.plot_eml_strength_map(
            eml_ds_time_big.eml_strength.isel(fulllevel=max_strength_ind)*100, 
            eml_ds_time_big.lat, eml_ds_time_big.lon, 
            eml_ds_time_big.time.values, 
            fig=fig, ax=ax2)
        # eml_ds_600 = eml_ds_time.where(
        #     (eml_ds_time.pfull > 50000) & 
        #     (eml_ds_time.pfull < 70000)).mean(
        #         'fulllevel')
        # uv_norm = np.sqrt(
        #     eml_ds_600.u[::10, ::10]**2 + eml_ds_600.v[::10, ::10]**2)
        # q = ax2.quiver(
        #         eml_ds_600.lon[::10], eml_ds_600.lat[::10], 
        #         eml_ds_600.u[::10, ::10]/uv_norm, 
        #         eml_ds_600.v[::10, ::10]/uv_norm, 
        #         scale=7,
        #         scale_units='inches',
        #         units='width',
        #         width=0.0008
        #         )
        # rr_levels = np.nanpercentile(
        # eml_ds_time.rain_rate*3600, [90, 100])
        # ax2.set(xlim=[-70, 20], ylim=[0, 30])

        # ax3 = fig.add_subplot(
        #     413, projection=ccrs.PlateCarree(), sharex=ax1)
        # fig, ax3 = eec.plot_eml_height_map(
        #     eml_ds_time_big.pfull.isel(lev=max_strength_ind)/100, 
        #     eml_ds_time_big.latitude, eml_ds_time_big.longitude, 
        #     eml_ds_time_big.time.values, fig=fig, ax=ax3)
        # # ax3.set_extent([-65, 0, 0, 25])
        # ax4 = fig.add_subplot(
        #     414, projection=ccrs.PlateCarree(), sharey=ax3)
        # fig, ax4 = eec.plot_eml_thickness_map(
        #     eml_ds_time_big.eml_pwidth.isel(lev=max_strength_ind)/100, 
        #     eml_ds_time_big.latitude, eml_ds_time_big.longitude, 
        #     eml_ds_time_big.time.values, fig=fig, ax=ax4) 
        # # ax4.set_extent([-65, 0, 0, 25])
        # ax1.set_anchor((0, 0.2))
        # # plt.subplots_adjust(hspace=-0.5)
        # plt.suptitle(f'{str(time)[:19]}', fontsize='8')
        # ax1.set_aspect(0.4)
        # ax2.set_aspect(0.4)
        # ax3.set_aspect(0.4)
        # ax4.set_aspect(0.4)
        plt.savefig(f'/home/u/u300676/moist-layers/plots/monsoon_gridded/'
                    f'monsoon_eml_strength_map_atlantic_{str(time)[:19]}.png',
                    dpi=300)
    eec.make_movie(
        image_paths='plots/era5/era5_composite_eml_map_*.png', 
        video_name='videos/era5_composite_eml_chars_video.mp4')

if __name__ == '__main__':
    main()
