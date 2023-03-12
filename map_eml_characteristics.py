import eval_eml_chars as eec
import moist_layers as ml
import intake
import verde as vd
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def main():
     # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask().sel(
        time=slice("2021-07-28T00:00:00", "2021-08-02T00:00:00"))
    ds2d = cat["luk1043"].atm2d.to_dask().sel(
        time=slice("2021-07-28T00:00:00", "2021-08-02T00:00:00"))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_eurec4a(ds3d, grid)
    ds2d = ml.mask_eurec4a(ds2d, grid)
    grid = ml.mask_eurec4a(grid, grid)
    times = ds3d.time.values
    for time in times:
        print(str(time)[:19])
        data3d = ds3d.sel(time=time)
        data2d = ds2d.sel(time=time)
        eml_ds = eec.load_eml_data(time)
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=1e-4, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
            
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(321, projection=ccrs.PlateCarree())
        fig, ax1 = eec.plot_q_map(
            data3d.hus.sel(fulllevel=68), grid.clat, grid.clon,
            fig=fig, ax=ax1)
        ax2 = fig.add_subplot(322, projection=ccrs.PlateCarree(), sharey=ax1)
        fig, ax2 = eec.plot_eml_strength_map(
            eml_ds.strength, np.rad2deg(eml_ds.lat), 
            np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax2)
        ax3 = fig.add_subplot(323, projection=ccrs.PlateCarree(), sharex=ax1)
        fig, ax3 = eec.plot_eml_height_map(
            eml_ds.pmean, np.rad2deg(eml_ds.lat), 
            np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax3)
        ax4 = fig.add_subplot(324, projection=ccrs.PlateCarree(), sharey=ax3)
        fig, ax4 = eec.plot_eml_thickness_map(
            eml_ds.pwidth, np.rad2deg(eml_ds.lat), 
            np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax4)
        ax5 = fig.add_subplot(325, projection=ccrs.PlateCarree())
        fig, ax5 = eec.map_rain_rates(
            data2d.rain_gsp_rate * 3600, 
            grid.clon, grid.clat, fig=fig, ax=ax5)
        plt.suptitle(f'{str(time)[:19]}')
        plt.tight_layout()
        plt.savefig(f'/home/u/u300676/moist-layers/plots/'
                    f'extended_composite_eml_map_{str(time)[:19]}.png')
    eec.make_movie(
        image_paths='plots/extended_composite_eml_map_*.png', 
        video_name='videos/extended_composite_eml_chars_video.mp4')

if __name__ == '__main__':
    main()