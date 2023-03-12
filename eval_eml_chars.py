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
    ax.set_extent([-65, 0, 5, 25], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 10), crs=ccrs.PlateCarree())
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
    ax.set_extent([-65, 0, 0, 25], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter() 
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 10), crs=ccrs.PlateCarree())
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
            vmin=100,
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
    ax.set_extent([-65, 0, 0, 25], crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
    # lat_formatter = LatitudeFormatter() 
    # ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -39, 10), crs=ccrs.PlateCarree())
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

def make_movie(image_paths, video_name):
    image_folder = 'plots'
    video_name = video_name

    images = glob.glob(image_paths)
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
    ax.set_extent([-65, 0, 0, 25], crs=ccrs.PlateCarree())
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
    return data.isel(eml_count=good)


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
        projection=pyproj.Proj(
            proj="merc", lat_ts=np.rad2deg(eml_ds.lat.mean().values)),
    )
    return mask

def get_eml_ds_for_zrange(eml_ds, zmin, zmax):
    layer_eml_ds = eml_ds.isel(
        eml_count=((eml_ds.zmean > zmin) & (eml_ds.zmean < zmax)))
    return layer_eml_ds

def random_color():
    levels = np.arange(0, 1, 0.001)
    return tuple(random.choice(levels) for _ in range(3))

def plot_eml_labels(eml_labels, lon, lat, cmap, 
                    fig=None, ax=None, relabel=False, max_eml_label=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    if relabel:
        plot_labels = eml_labels.copy()
        plot_label = 0
        for label in np.unique(eml_labels):
            plot_labels[plot_labels==label] = plot_label
            plot_label += 1
    else:
        plot_labels = eml_labels
    s = ax.scatter(lon, lat, c=plot_labels, s=1, cmap=cmap, 
                   vmin=0, vmax=max_eml_label, transform=ccrs.PlateCarree())
    cb = plt.colorbar(s, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    cb.ax.set_xlabel("EML label", fontsize=10)
    cb.ax.tick_params(labelsize=10)
    cb.ax.set_xlim([0, max_eml_label])
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -0, 0, 25], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 26, 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -1, 10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    return fig, ax

def load_eml_data(time):
    eml_ds = xr.open_dataset(
        f'eml_data/eml_chars_extended_{str(time)[:19]}.nc') 
    eml_ds = eml_ds.assign_coords(
            {'eml_count': np.arange(len(eml_ds.eml_count))})
    return eml_ds

def get_3d_height_eml_labels(eml_ds, grid, spacing, heights):
    eml_mask_3d = np.zeros((grid[0].shape + (len(heights),)))
    for i, height in enumerate(heights):
        layer_eml_ds = get_eml_ds_for_zrange(eml_ds, height, height+100)
        if len(layer_eml_ds.eml_count) == 0: 
            continue
        eml_mask = get_eml_mask(
            layer_eml_ds, grid=grid, maxdist=spacing*1*111e3)
        eml_mask_3d[:, :, i] = eml_mask
    eml_labels_3d = label(xr.where(eml_mask_3d, 1, 0), background=0)
    return eml_labels_3d

def project_3d_labels_to_2d(labels3d):
    props = np.array(regionprops(labels3d))
    area_sort_ind = np.argsort([prop.area for prop in props])
    sorted_labels = [prop.label for prop in props[area_sort_ind]]
    labels2d = np.zeros(labels3d[:, :, 0].shape, dtype=np.int64)
    for label in sorted_labels[::-1]: # Go from big to small objects
        label3d_proj_bool = np.any(labels3d == label, axis=2)
        labels2d[label3d_proj_bool] = label
    return labels2d

def get_3d_time_eml_labels(times, grid, spacing):
    eml_mask_3d = np.zeros((grid[0].shape + (len(times),)))
    for i, time in enumerate(times):
        print(f'{str(time)[:19]}')
        eml_ds = load_eml_data(time)
        eml_ds = filter_eml_data(
            eml_ds, 
            min_strength=1e-4, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
        eml_mask = get_eml_mask(eml_ds, grid=grid, maxdist=spacing*1*111e3)
        eml_labels = label(xr.where(eml_mask, 1, 0), background=0)
        eml_props = regionprops(eml_labels)
        eml_areas = [props.area for props in eml_props] 
        big_emls = [props.label for props in eml_props 
                    if props.area > 3000]
        big_eml_mask = xr.where(np.isin(eml_labels, big_emls), 1, 0)
        eml_mask_3d[:, :, i] = big_eml_mask
    eml_labels_3d = label(xr.where(eml_mask_3d, 1, 0), background=0)
    return eml_labels_3d, grid


def map_rain_rates(rr, lon, lat, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cs1 = ax.scatter(
            np.rad2deg(lon),
            np.rad2deg(lat),
            s=1,
            c=rr,
            vmin=0,
            vmax=50,
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
    cb1.ax.set_xlabel("rain rate (kg m$^{-2}$)", fontsize=10)
    cb1.ax.tick_params(labelsize=10)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([-65, -0, 0, 25], crs=ccrs.PlateCarree())
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


def calc_updraft_height(wvec, zvec):
    zuh_ind = np.where(wvec <= 0)[0].max()
    if zuh_ind == 89:
        return 0.
    else:
        return zvec[zuh_ind-1:zuh_ind+1].mean()

def assign_updraft_height(data):
    return data.assign(
        {
            'updraft_height': (
                ('cell'), 
                [calc_updraft_height(
                    data.wa[:-1, cell], 
                    data.zg[:-1, cell]) 
                    for cell in range(len(data.cell))
                    ])  
        }
    )

def main():
    pass
         
if __name__ == '__main__':
    main()