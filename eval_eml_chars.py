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
from scipy.interpolate import griddata
from skimage.measure import label
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
import verde as vd
import pyproj
import random
from matplotlib.colors import ListedColormap

from typhon.physics import (
    vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk)
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
            vmin=1e-4,
            vmax=0.005,
            cmap="density",
            transform=ccrs.PlateCarree(),
        )
    ax.plot([-45, -45], [8, 25],
             color='red', linestyle='-',
             transform=ccrs.PlateCarree(),
             )
    ax.scatter(
        -45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
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
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(
        cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
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
            c=eml_height/100,
            vmin=500,
            vmax=700,
            cmap="density",
            transform=ccrs.PlateCarree(),
        )
    ax.plot([-45, -45], [8, 25],
             color='red', linestyle='-',
             transform=ccrs.PlateCarree(),
             )
    ax.scatter(-45, 21, color=sns.color_palette(
        'colorblind')[2], marker='x', s=100)
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
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(
        cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    cb.ax.set_xlabel(r"EML height / hPa")
    cb.ax.tick_params(labelsize=10)

    return fig, ax

def plot_eml_thickness_map(eml_pwidth, lat, lon, time, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cs1 = ax.scatter(
            lon,
            lat,
            s=1,
            c=eml_pwidth/100,
            vmin=50,
            vmax=300,
            cmap="density",
            transform=ccrs.PlateCarree(),
        )
    ax.plot([-45, -45], [8, 25],
             color='red', linestyle='-',
             transform=ccrs.PlateCarree(),
             )
    ax.scatter(-45, 21, color=sns.color_palette(
        'colorblind')[2], marker='x', s=100)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -40, 5, 25], crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
    # lat_formatter = LatitudeFormatter() 
    # ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(
        cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    cb.ax.set_xlabel(r"EML thickness / hPa")
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
    video_name = 'big_eml_label_video.mp4'

    images = glob.glob("plots/big_eml_labels_*.png")
    images.sort()
    # images.sort(
    # key=lambda image: float(re.findall('[+-]?[0-9]?[0-9]', image)[0]))
    frame = cv2.imread(images[0])
    height, width, layers= frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width,height))

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
    ax.scatter(
        -45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    cb1 = plt.colorbar(
        cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    cb1.ax.set_xlabel("500 hPa Specific Humidity (kg/kg)", fontsize=10)
    cb1.ax.tick_params(labelsize=10)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -40, 5, 25], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    # ax.set_xticks(np.arange(-65, -39, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    return fig, ax
    
def filter_eml_data(
    data, min_strength, min_pmean, max_pmean, min_pwidth, max_pwidth):
    good = (
        (data.strength > min_strength) &
        (data.pmean > min_pmean) &
        (data.pmean < max_pmean) &
        (data.pwidth > min_pwidth) &
        (data.pwidth < max_pwidth)
    )
    return data.where(good, drop=True)


def grid_eml_data(data):
    lon = np.rad2deg(data.lon.values)
    lat = np.rad2deg(data.lat.values)
    new_lon = np.arange(-65, -40, 0.1)
    new_lat = np.arange(5, 25, 0.1)
    new_mgrid = np.meshgrid(new_lon, new_lat)
    new_lon_mgrid = new_mgrid[0]
    new_lat_mgrid = new_mgrid[1]
    gridded_ds = xr.Dataset(
        coords=
        {
            'lat': new_lat,
            'lon': new_lon,
            'time': data.time[0],
        },
        data_vars=
        {
            f'{var}': (('lon', 'lat'), 
                griddata(
                    (lon, lat), data[f'{var}'], 
                    (new_lon_mgrid, new_lat_mgrid), method='nearest').T)
                for var in data.data_vars
                if var not in ('lat', 'lon', 'time')
        }
    )
    return gridded_ds


def label_eml_field(gridded_ds):
    eml_mask = xr.where(gridded_ds.strength.isnull(), x=0, y=1)
    eml_labels = label(eml_mask, 0)
    return eml_labels

def get_eml_mask(eml_ds, grid, maxdist):
    mask = vd.distance_mask(
        (np.rad2deg(eml_ds.lon), np.rad2deg(eml_ds.lat)),
        maxdist=maxdist,
        coordinates=grid,
        projection=pyproj.Proj(proj="merc", lat_ts=np.rad2deg(eml_ds.lat.mean().values)),
    )
    return mask

def random_color():
    levels = np.arange(0, 1, 0.001)
    return tuple(random.choice(levels) for _ in range(3))

def plot_eml_labels(eml_labels, lon, lat, cmap, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    plot_labels = eml_labels.copy()
    plot_label = 1
    for label in np.unique(eml_labels):
        plot_labels[plot_labels==label] = plot_label
        plot_label += 1
    s = ax.scatter(lon, lat, c=plot_labels, s=1, cmap=cmap, 
                    transform=ccrs.PlateCarree())
    cb = plt.colorbar(s, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    cb.ax.set_xlabel("EML label", fontsize=10)
    cb.ax.tick_params(labelsize=10)
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
    random.seed(42)
    colors = [random_color() for i in range(500)]
    colors[0] = (1, 1, 1)
    for time in times:
        print(str(time)[:19])
        data = ds3d.sel(time=time)
        eml_ds = xr.open_dataset(f'eml_data/eml_chars_{str(time)[:19]}.nc') 
        eml_ds = eml_ds.assign_coords(
            {'eml_count': np.arange(len(eml_ds.eml_count))})
        eml_ds = filter_eml_data(
            eml_ds, 
            min_strength=1e-4, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
        region = vd.get_region(
            (np.rad2deg(eml_ds.lon.values), np.rad2deg(eml_ds.lat.values)))
        spacing = 0.05
        grid = vd.grid_coordinates(region, spacing=spacing)
        eml_mask = get_eml_mask(eml_ds, grid=grid, maxdist=spacing*1*111e3)
        eml_labels = label(xr.where(eml_mask, 1, 0), background=0)
        eml_props = regionprops(eml_labels)
        eml_areas = [props.area for props in eml_props] 
        big_emls = [props.label for props in eml_props 
                    if props.area > np.mean(eml_areas)]
        eml_labels = xr.where(np.isin(eml_labels, big_emls), eml_labels, 0)

        fig = plt.figure(figsize=(9, 4.5))
        ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
        fig, ax1 = plot_eml_strength_map(
            eml_ds.strength, np.rad2deg(eml_ds.lat), 
            np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax1)
        ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
        cmap = ListedColormap(colors[:len(np.unique(eml_labels))], 
                              name='eml_label_colors')
        fig, ax2 = plot_eml_labels(eml_labels, grid[0], grid[1], cmap,
                                   fig=fig, ax=ax2)
        plt.suptitle(f'{str(time)[:19]}')
        plt.savefig(f'plots/big_eml_labels_{str(time)[:19]}.png')
        
    #     fig = plt.figure(figsize=(9, 9))
    #     fig, ax1 = plot_q_map(data.hus.sel(fulllevel=68), grid.clat, grid.clon,
    #                           fig=fig, ax=ax1)
    #     ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree(), sharey=ax1)
    #     fig, ax2 = plot_eml_strength_map(
    #         eml_ds.strength, np.rad2deg(eml_ds.lat), 
    #         np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax2)
    #     ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree(), sharex=ax1)
    #     fig, ax3 = plot_eml_height_map(
    #         eml_ds.pmean, np.rad2deg(eml_ds.lat), 
    #         np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax3)
    #     ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree(), sharey=ax3)
    #     fig, ax4 = plot_eml_thickness_map(
    #         eml_ds.pwidth, np.rad2deg(eml_ds.lat), 
    #         np.rad2deg(eml_ds.lon), eml_ds.time[0].values, fig=fig, ax=ax4)
    #     plt.suptitle(f'{str(time)[:19]}')
    #     plt.tight_layout()
    #     plt.savefig(f'/home/u/u300676/moist-layers/plots/'
    #                 f'composite_eml_map_{str(time)[:19]}.png')
    # make_movie()
if __name__ == '__main__':
    main()