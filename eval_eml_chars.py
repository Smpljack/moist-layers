import numpy as np
from sympy import true
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import intake
import cv2
import glob

from typhon.physics import vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk
from typhon.plots import profile_p

import moist_layers as ml


def plot_eml_strength_map(eml_strength, lat, lon, time, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cs1 = ax.scatter(
            lon,
            lat,
            s=1,
            c=eml_strength,
            vmin=0,
            # vmax=0.005,
            cmap="Blues",
            transform=ccrs.PlateCarree(),
        )
    ax.plot([-45, -45], [8, 25],
             color='red', linestyle='-',
             transform=ccrs.PlateCarree(),
             )
    ax.scatter(-45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -40, 5, 25], crs=ccrs.PlateCarree())
    # ax.set_yticks([], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    cb.ax.set_xlabel(r"EML strength")
    cb.ax.tick_params(labelsize=10)

    return fig, ax

def plot_eml_height_map(eml_height, lat, lon, time, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cs1 = ax.scatter(
            lon,
            lat,
            s=1,
            c=eml_height,
            vmin=40000,
            vmax=70000,
            cmap="Reds",
            transform=ccrs.PlateCarree(),
        )
    ax.plot([-45, -45], [8, 25],
             color='red', linestyle='-',
             transform=ccrs.PlateCarree(),
             )
    ax.scatter(-45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -40, 5, 25], crs=ccrs.PlateCarree())
    # ax.set_yticks([], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    cb.ax.set_xlabel(r"EML height")
    cb.ax.tick_params(labelsize=10)

    return fig, ax

def add_rh_to_ds(data):
    return data.assign(
        {'rh': (
            ('cell', 'time'), 
            vmr2relative_humidity(
                specific_humidity2vmr(data.hus),
                data.pfull,
                data.ta,
                e_eq=e_eq_mixed_mk)
            )}
    )

def plot_rh_anom_xsec(data, grid, mean_data):
    
    data_cs = ml.mask_cross_section(data, grid)
    data_cs = add_rh_to_ds(data)
    sort_lat = np.argsort(data.clat)
    data_cs = data.sortby(sort_lat)
    data_point = ml.mask_point()

def make_movie():
    image_folder = 'plots'
    video_name = 'eml_video_new.mp4'

    images = glob.glob("plots/composite_eml_map*.png")
    images.sort()
    # images.sort(key=lambda image: float(re.findall('[+-]?[0-9]?[0-9]', image)[0]))
    frame = cv2.imread(images[0])
    height, width, layers= frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def plot_q_map(q, lat, lon, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cs1 = ax.scatter(
            np.rad2deg(lon),
            np.rad2deg(lat),
            s=1,
            c=q,
            vmin=0,
            vmax=0.005,
            cmap="density",
            transform=ccrs.PlateCarree(),
        )
    ax.plot([-45, -45], [8, 25],
             color='red', linestyle='-',
             transform=ccrs.PlateCarree(),
             )
    ax.scatter(-45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    cb1 = plt.colorbar(cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    cb1.ax.set_xlabel("500 hPa Specific Humidity (kg/kg)", fontsize=10)
    cb1.ax.tick_params(labelsize=10)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -40, 5, 25], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    return fig, ax
    
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
    for time in times:
        print(str(time)[:19])
        eml_ds = xr.open_dataset(f'eml_data/eml_chars_{str(time)[:19]}.nc') 
        eml_ds = eml_ds.assign_coords(
            {'eml_count': np.arange(len(eml_ds.eml_count))})
        data = ds3d.sel(time=time)
        fig = plt.figure(figsize=(9,9))
        ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
        fig, ax1 = plot_q_map(data.hus.sel(fulllevel=68), grid.clat, grid.clon,
                              fig=fig, ax=ax1)
        ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree(), sharey=ax1)
        fig, ax2 = plot_eml_strength_map(eml_ds.strength, np.rad2deg(eml_ds.lat), 
                              np.rad2deg(eml_ds.lon), eml_ds.time[0].values, 
                              fig=fig, ax=ax2)
        ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree(), sharex=ax1)
        fig, ax3 = plot_eml_height_map(eml_ds.pmean, np.rad2deg(eml_ds.lat), 
                              np.rad2deg(eml_ds.lon), eml_ds.time[0].values, 
                              fig=fig, ax=ax3)
        plt.savefig(f'/home/u/u300676/moist-layers/plots/'
                    f'composite_eml_map_{str(time)[:19]}.png')

if __name__ == '__main__':
    main()