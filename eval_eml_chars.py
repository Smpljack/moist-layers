import numpy as np
from sympy import true
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
# import intake
# import cv2
import glob
from scipy.interpolate import griddata
from skimage.measure import label
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
# import verde as vd
import pyproj
import random
from matplotlib.colors import ListedColormap
import metpy.calc as mpcalc
import matplotlib 
import matplotlib.patches as mpatches


from typhon.physics import (
    vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk)
from typhon.plots import profile_p

# import moist_layers as ml

def filter_gridded_eml_ds(
    eml_ds,  min_strength, min_pmean, max_pmean, min_pwidth, max_pwidth):
    eml_vars = [var for var in eml_ds.variables if var[:3] == 'eml']
    for eml_var in eml_vars:
        eml_ds_prange = eml_ds.where(
            (eml_ds.pfull > min_pmean) &
            (eml_ds.pfull < max_pmean)
        )
        eml_ds[eml_var] = eml_ds[eml_var].where(
            np.any(eml_ds_prange.eml_strength > min_strength, axis=0) &
            np.any(eml_ds_prange.eml_pwidth < max_pwidth, axis=0) &
            np.any(eml_ds_prange.eml_pwidth > min_pwidth, axis=0)
        )
    return eml_ds

def filter_gridded_eml_ds_all_vars(
    eml_ds,  min_strength, min_pmean, max_pmean, min_pwidth, max_pwidth):

    eml_ds_prange = eml_ds.where(
        (eml_ds.pfull > min_pmean) &
        (eml_ds.pfull < max_pmean)
    )
    eml_ds = eml_ds.where(
        np.any(eml_ds_prange.eml_strength > min_strength, axis=0) &
        np.any(eml_ds_prange.eml_pwidth < max_pwidth, axis=0) &
        np.any(eml_ds_prange.eml_pwidth > min_pwidth, axis=0)
    )
    return eml_ds

def plot_eml_strength_map(eml_strength, lat, lon, time, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    if len(lon) != len(lat):
        cs1 = ax.pcolormesh(
            lon, lat, eml_strength, vmin=30, vmax=70, cmap='density',
            transform=ccrs.PlateCarree())
    else:
        cs1 = ax.scatter(
                lon,
                lat,
                s=0.3,
                c=eml_strength,
                vmin=30,
                vmax=50,
                cmap="density",
                transform=ccrs.PlateCarree(),
            )
    # ax.plot([-45, -45], [8, 25],
    #          color='red', linestyle='-',
    #          transform=ccrs.PlateCarree(),
    #          )
    # ax.scatter(
    #     -45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    ax.coastlines(resolution="10m", linewidth=1.5)
    # ax.set_yticks(np.arange(lat.min(), lat.max()+1, 15))
    # lat_formatter = LatitudeFormatter() 
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.set_xticks(np.arange(lon.min(), lon.max(), 10),)
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.tick_params(labelsize=25)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # ax.set_aspect(1)
    # ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    # ax.set_title(f"{str(time)[:19]}")
    # cb = plt.colorbar(
    #     cs1, extend="max", orientation="vertical", shrink=0.7, pad=0.04, aspect=10)
    # cb.ax.set(xticks=[-3, -2], xticklabels=[r'10$^{-3}$', r'10$^{-2}$'])
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    # cb.ax.set_ylabel("EML strength / %", fontsize=4)
    # cb.ax.tick_params(labelsize=4)

    return fig, ax

def plot_eml_height_map(eml_height, lat, lon, time, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    if len(lon) != len(lat):
        cs1 = ax.pcolormesh(
            lon, lat, eml_height, vmin=500, vmax=800, cmap='density',
            transform=ccrs.PlateCarree())
    else:
        cs1 = ax.scatter(
                lon,
                lat,
                s=0.3,
                c=eml_height,
                vmin=500,
                vmax=800,
                cmap="density",
                transform=ccrs.PlateCarree(),
            )
    # ax.plot([-45, -45], [8, 25],
    #          color='red', linestyle='-',
    #          transform=ccrs.PlateCarree(),
    #          )
    # ax.scatter(-45, 21, color=sns.color_palette(
    #     'colorblind')[2], marker='x', s=100)
    ax.coastlines(resolution="10m", linewidth=0.6)
    # ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(lat.min(), lat.max(), 5), crs=ccrs.PlateCarree())
    # lat_formatter = LatitudeFormatter() 
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.set_xticks(np.arange(lon.min(), lon.max(), 10), crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(
        cs1, extend="max", orientation="vertical", shrink=0.7, pad=0.04, aspect=10)
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    cb.ax.set_ylabel(r"EML height / hPa", fontsize=4)
    cb.ax.tick_params(labelsize=4)

    return fig, ax

def plot_eml_thickness_map(eml_pwidth, lat, lon, time, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    if len(lon) != len(lat):
        cs1 = ax.pcolormesh(
            lon, lat, eml_pwidth, vmin=50, vmax=400, cmap='density',
            transform=ccrs.PlateCarree())
    else:
        cs1 = ax.scatter(
                lon,
                lat,
                s=0.3,
                c=eml_pwidth,
                vmin=50,
                vmax=400,
                cmap="density",
                transform=ccrs.PlateCarree(),
            )
    # ax.plot([-45, -45], [8, 25],
    #          color='red', linestyle='-',
    #          transform=ccrs.PlateCarree(),
    #          )
    # ax.scatter(-45, 21, color=sns.color_palette(
    #     'colorblind')[2], marker='x', s=100)
    ax.coastlines(resolution="10m", linewidth=0.6)
    # ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(lat.min(), lat.max(), 5), crs=ccrs.PlateCarree())
    # lat_formatter = LatitudeFormatter() 
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.set_xticks(np.arange(lon.min(), lon.max(), 10), crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    # ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(
        cs1, extend="max", orientation="vertical", shrink=0.7, pad=0.04, aspect=10)
    cb.ax.set_ylabel(r"EML thickness / hPa", fontsize=4)
    cb.ax.tick_params(labelsize=4)

    return fig, ax


def add_rh_to_ds(data, out_dims=('cell', 'time')):
    return data.assign(
        {'rh': (
            out_dims, 
            vmr2relative_humidity(
                specific_humidity2vmr(data.hus).values,
                data.pfull.values,
                data.ta.values,
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
    import re
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

def plot_gridded_q_map(q, lat, lon, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cs1 = ax.pcolormesh(
            lon,
            lat,
            q,
            vmin=0,
            vmax=0.005,
            cmap="density",
            transform=ccrs.PlateCarree(),
        )
    # ax.plot([-45, -45], [8, 25],
    #          color='red', linestyle='-',
    #          transform=ccrs.PlateCarree(),
    #          )
    # ax.scatter(
    #     -45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    cb1 = plt.colorbar(
        cs1, extend="max", orientation="vertical", shrink=0.7, pad=0.04, aspect=10, 
        # anchor=(0.1, 1.5)
        )
    cb1.ax.set_ylabel("Specific Humidity\n500hPa / kg kg$^{-1}$)", fontsize=4)
    cb1.ax.tick_params(labelsize=4)
    ax.coastlines(resolution="10m", linewidth=0.6)
    # ax.set_yticks(np.arange(lat.min(), lat.max(), 5))
    # lat_formatter = LatitudeFormatter()
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.set_xticks(np.arange(lon.min(), lon.max(), 10))
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=5)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    # ax.set_extent(
    #     [160, -140, lat.min(), lat.max()])
    return fig, ax

def plot_q_map(q, lat, lon, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    lon = np.rad2deg(lon),
    lat = np.rad2deg(lat), 
    cs1 = ax.scatter(
            lon, 
            lat,
            s=0.3,
            c=q,
            vmin=0,
            vmax=0.005,
            cmap="density",
            transform=ccrs.PlateCarree(),
        )
    # ax.plot([-45, -45], [8, 25],
    #          color='red', linestyle='-',
    #          transform=ccrs.PlateCarree(),
    #          )
    # ax.scatter(
    #     -45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
    cb1 = plt.colorbar(
        cs1, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    cb1.ax.set_xlabel("500 hPa Specific Humidity (kg/kg)", fontsize=10)
    cb1.ax.tick_params(labelsize=10)
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat.min(), lat.max(), 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter() 
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(lon.min(), lon.max(), 10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    return fig, ax

def plot_rr_contour(rr, lat, lon, levels, fig, ax):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    cf = ax.contourf(
        lon, lat, rr, 
        levels=levels, 
        cmap='Reds',
        # vmin=levels[0]*1.1,
        # vmax=levels[2]*0.9,
        extend='max',
        # colors=[(0.3, 0.3, 0.3), 
        #         (0.6, 0.6, 0.6),
        #         (1, 1, 1)],
        transform=ccrs.PlateCarree(),
    )
    cb1 = plt.colorbar(
        cf, orientation="vertical", shrink=0.9, pad=0.01, 
        # anchor=(0, 0.2)
        )
    cb1.ax.set_ylabel(r"rain rate / mm day$^{-1}$", fontsize=7)
    cb1.ax.set_yticks(levels)
    cb1.ax.set_yticklabels([str(level) for level in np.round(levels, 2)]) 
    # cb1.ax.set_ylim([levels[0]-0.1, levels[2]+0.1])
    cb1.ax.tick_params(labelsize=7)
    
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

def grid_monsoon_data(data, var_name, lat_grid, lon_grid):
    lon, lat = np.meshgrid(lon_grid, lat_grid)
    gridded_data = griddata(
        values=data[f"{var_name}"],
        points=(data.clon, data.clat),
        xi=(lon, lat),
    )
    return gridded_data

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

def relabel(labels2d):
    new_labels = labels2d.copy()
    for new_label, label in enumerate(np.unique(labels2d)):
        new_labels[new_labels==label] = new_label
    return new_labels

def plot_eml_labels(eml_labels, lon, lat, cmap, 
                    fig=None, ax=None, max_eml_label=None):
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())  
    if len(lon) != len(lat):
        s = ax.pcolormesh(
            lon, lat, eml_labels, vmin=0, vmax=max_eml_label, cmap=cmap,
            transform=ccrs.PlateCarree())
    else:
        s = ax.scatter(lon, lat, c=eml_labels, s=0.3, cmap=cmap, 
                    vmin=0, vmax=max_eml_label, transform=ccrs.PlateCarree())
    # cb = plt.colorbar(s, extend="max", orientation="horizontal", shrink=0.8, pad=0.1)
    # cb.ax.set_xlabel("EML label", fontsize=10)
    # cb.ax.tick_params(labelsize=10)
    # cb.ax.set_xlim([0, max_eml_label])
    ax.coastlines(resolution="110m", linewidth=1)
    # ax.set_yticks(np.arange(0, 26, 5), crs=ccrs.PlateCarree())
    # lat_formatter = LatitudeFormatter()
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.set_xticks(np.arange(-65, -1, 10), crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_aspect(1)
    return fig, ax

def load_eml_data(time, data_path_keyword):
    eml_ds = xr.open_dataset(
        '/home/u/u300676/user_data/mprange/eml_data/'
        f'eml_chars_{data_path_keyword}_{str(time)[:19]}.nc') 
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

def get_3d_height_eml_labels_griddata(
    eml_ds, lat, lon, heights, height_tolerance=100):
    eml_mask_3d = np.zeros((len(lat), len(lon), len(heights)))
    for i, height in enumerate(heights):
        eml_mask = xr.where(
                np.any(
                    eml_ds.eml_zmean > height - height_tolerance/2, axis=0) & 
                np.any(
                    eml_ds.eml_zmean < height + height_tolerance/2, axis=0), 
                    x=True, y=False)
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
            s=0.3,
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
    ax.set_yticks(np.arange(0, 26, 5), crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(-65, -0, 10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
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


def get_eml_chars_for_label(eml_ds, eml_labels, eml_label_grid, label, 
                            eml_ds_lat_lon_tuples=None):
    eml_grid_ind = (eml_labels == label)
    eml_grid_lons = eml_label_grid[0][eml_grid_ind]
    eml_grid_lats = eml_label_grid[1][eml_grid_ind]
    label_points = [(lat, lon) for lat, lon in zip(eml_grid_lats, eml_grid_lons)]
    if eml_ds_lat_lon_tuples is None:
        eml_ds_lat_lon_tuples = [(lat, lon) for lat, lon 
                    in zip(np.rad2deg(eml_ds.lat.values), 
                            np.rad2deg(eml_ds.lon.values))]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
                eml_ds_lat_lon_tuples)
    distances, indices = nbrs.kneighbors(label_points)
    return eml_ds.isel(eml_count=indices[:, 0]).expand_dims('eml_label')

def get_eml_chars_subset_for_labels(eml_ds, eml_labels, eml_label_grid, 
                                    eml_ds_lat_lon_tuples=None):
    eml_ds['eml_count'] = np.arange(len(eml_ds.eml_count))
    eml_ds_subset_labels = None
    if eml_ds_lat_lon_tuples is None:
        eml_ds_lat_lon_tuples = [
            (lat, lon) for lat, lon in zip(
                np.rad2deg(eml_ds.lat.values), np.rad2deg(eml_ds.lon.values))
            ]
    for label in np.unique(eml_labels):
        if label == 0:
            continue
        eml_ds_subset_label = get_eml_chars_for_label(
            eml_ds, eml_labels, eml_label_grid, label, eml_ds_lat_lon_tuples).squeeze()
        if eml_ds_subset_labels is None:
            eml_ds_subset_labels = eml_ds_subset_label
        else:
            eml_ds_subset_labels = xr.concat(
                [eml_ds_subset_labels, eml_ds_subset_label], dim='eml_count')
    return eml_ds_subset_labels
    
def get_ds3d_subset_for_label(ds3d, eml_labels, eml_label_grid, label, 
                              ds3d_lat_lon_tuples=None):
    eml_grid_ind = (eml_labels == label)
    eml_grid_lons = eml_label_grid[0][eml_grid_ind]
    eml_grid_lats = eml_label_grid[1][eml_grid_ind]
    label_points = [(lat, lon) for lat, lon in zip(eml_grid_lats, eml_grid_lons)]
    if ds3d_lat_lon_tuples is None:
        ds3d_lat_lon_tuples = [(lat, lon) for lat, lon 
                    in zip(np.rad2deg(ds3d.clat.values), 
                            np.rad2deg(ds3d.clon.values))]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
                ds3d_lat_lon_tuples)
    distances, indices = nbrs.kneighbors(label_points)
    return ds3d.isel(cell=indices[:, 0]).expand_dims('eml_label')

def get_ds3d_subset_for_labels(ds3d, eml_labels, eml_label_grid, 
                               ds3d_lat_lon_tuples=None):
    ds3d_subset_labels = None
    if ds3d_lat_lon_tuples is None:
        ds3d_lat_lon_tuples = [
            (lat, lon) for lat, lon in zip(
                np.rad2deg(ds3d.clat.values), np.rad2deg(ds3d.clon.values))
            ]
    for label in np.unique(eml_labels):
        if label == 0:
            continue
        ds3d_subset_label = get_ds3d_subset_for_label(
            ds3d, eml_labels, eml_label_grid, label, ds3d_lat_lon_tuples).squeeze()
        if ds3d_subset_labels is None:
            ds3d_subset_labels = ds3d_subset_label
        else:
            ds3d_subset_labels = xr.concat(
                [ds3d_subset_labels, ds3d_subset_label], dim='cell')
    return ds3d_subset_labels

def get_eml_char_ds_for_labelled_emls(eml_labels2d, label_grid, eml_ds, ds3d=None):
    eml_regionprops = regionprops(eml_labels2d)
    eml_label_ds = xr.Dataset(
        coords={
            'eml_label': np.array(
                [props.label for props in eml_regionprops])
        },
        data_vars={
            f'{char}_mean': (
                ('eml_label'), 
                np.full(len(eml_regionprops), np.nan))
            for char in ['strength', 'pmean', 'pwidth', 'zmean', 'zwidth']
            }
        )
    eml_label_ds = eml_label_ds.assign(
        variables={
            'area': (
                ('eml_label'), 
                np.array([props.area for props in eml_regionprops])),
            })
    eml_ds_lat_lon_tuples = [
        (lat, lon) for lat, lon in zip(
            np.rad2deg(eml_ds.lat.values), np.rad2deg(eml_ds.lon.values))
        ]
    for i, props in enumerate(eml_regionprops):
        mean_eml_chars_label = get_eml_chars_for_label(
            eml_ds, eml_labels2d, label_grid, props.label, eml_ds_lat_lon_tuples,
        ).mean('eml_count')

        for char in ['strength', 'pmean', 'pwidth', 'zmean', 'zwidth']:
            eml_label_ds[f'{char}_mean'][i] = mean_eml_chars_label[f'{char}'][0]
    return eml_label_ds

def get_mean_profiles_for_labelled_emls_griddata(
    eml_labels2d, eml_ds, vertical_dim):
    eml_regionprops = regionprops(eml_labels2d)
    eml_label_ds = xr.Dataset(
        coords={
            'eml_label': np.array(
                [props.label for props in eml_regionprops]),
            vertical_dim: eml_ds[vertical_dim].values,
        },
        data_vars={
            f'{var}_mean': (
                ('eml_label', vertical_dim), 
                np.full(
                    (len(eml_regionprops), len(eml_ds[vertical_dim].values)), 
                    np.nan))
            for var in ['pfull', 't', 'q', 'z', 'rh', 'hus', 'ta', 'zg'] 
            if var in eml_ds.variables
            }
        )
    for i, props in enumerate(eml_regionprops):
        mean_eml_profiles = eml_ds.where(
            eml_labels2d == props.label
        ).mean(['lat', 'lon'])

        for var in ['pfull', 't', 'q', 'z', 'rh', 'hus', 'ta', 'zg']:
            if var in eml_ds.variables:
                eml_label_ds[f'{var}_mean'][i, :] = \
                    mean_eml_profiles[f'{var}'].load()
    return eml_label_ds
    
def hovmoller_plot(
    lons, times, data_fill, lat_min, lat_max, title, data_line=None,):
    fig = plt.figure(figsize=(10, 13))
    # Use gridspec to help size elements of plot; small top plot and big bottom plot
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 5], hspace=0.1)

    # Tick labels
    x_tick_labels = [u'180\N{DEGREE SIGN}W', u'90\N{DEGREE SIGN}W',
                     u'60\N{DEGREE SIGN}W', u'0\N{DEGREE SIGN}',
                     u'20\N{DEGREE SIGN}E',
                     u'90\N{DEGREE SIGN}E', u'180\N{DEGREE SIGN}E']

    # Top plot for geographic reference (makes small map)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    ax1.hlines([lat_min, lat_max], lons.min(), lons.max(),
     colors=['red', 'red'], linestyles=['--', '--'])
    ax1.set_extent([-180, 180, -30, 30], 
        ccrs.PlateCarree(central_longitude=0))
    ax1.set_yticks([-30, -15, 0, 15, 30])
    ax1.set_yticklabels(
        [u'30\N{DEGREE SIGN}S', u'15\N{DEGREE SIGN}S', u'Eq', 
         u'15\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N'])
    ax1.set_xticks([-180, -90, -60, 0, 20, 90, 180])
    ax1.set_xticklabels(x_tick_labels)
    ax1.grid(linestyle='dotted', linewidth=2)

    # Add geopolitical boundaries for map reference
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
    # ax1.add_feature(
    #     cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

    # Set some titles
    plt.title('Hovmoller Diagram', loc='left')
    plt.title(title, loc='right')

    # Bottom plot for Hovmoller diagram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.invert_yaxis()  # Reverse the time order to do oldest first

    # Plot of chosen variable averaged over latitude and slightly smoothed
    clevs_line = [-1000, 0, 1000]#np.arange(-2, 4, 2)
    clevs_fill = np.arange(30, 60, 2)
    # Tick labels
    x_tick_labels = [
        u'60\N{DEGREE SIGN}W', u'40\N{DEGREE SIGN}W', u'20\N{DEGREE SIGN}W', 
        u'0\N{DEGREE SIGN}W', u'20\N{DEGREE SIGN}E'
        ]
    x_ticks = [-60, -40, -20, 0, 20]
    # cf = ax2.contourf(
    #     lons, times, data_fill, clevs_fill, 
    #     cmap='density', extend='both', linestyles='solid',
    #     negative_linestyles='dashed')
    cf = ax2.pcolormesh(lons, times, data_fill, cmap='Blues', vmin=30, vmax=60)
    cs = ax2.contourf(
        lons, times, mpcalc.smooth_n_point(data_line, 9, 2), 
        clevs_line, hatches=['/', '\\'], alpha=0, linewidths=1.5,
        colors='k', #negative_linestyles='dashed',
        )
    # cs = ax2.squiver(lons[::8], times[::4], data_line['ua'][::4, ::8], data_line['va'][::4, ::8])
    cs = ax2.contour(
        lons, times, mpcalc.smooth_n_point(
        data_line, 9, 2), [0], linewidths=2.5, alpha=1, 
        colors='k',
        )
    cax1 = ax2.inset_axes([1.02, 0, 0.04, 1])
    cbar1 = plt.colorbar(
        cf, orientation='vertical', aspect=50, extendrect=True, 
        cax=cax1)
    cbar1.set_label('EML strength / %', fontsize=18)
    cbar1.ax.tick_params(labelsize=18)
    patch1 = mpatches.Patch(alpha=1, hatch=r'////', label='v < 0', facecolor='white')
    patch2 = mpatches.Patch(alpha=1, hatch=r'\\\\', label='v > 0', facecolor='white')

    # norm = matplotlib.colors.Normalize(
    #     vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    # sm.set_array([])
    # cax2 = ax2.inset_axes([0, -0.3, 1, 0.04])
    # cbar2 = plt.colorbar(
    #     sm, orientation='horizontal', pad=0.01, aspect=50,
    #     cax=cax2, extendrect=True,)
    # cbar2.set_label('v-wind / ms$^{-1}$', fontsize=18)
    # cbar2.ax.tick_params(labelsize=18)
    # Make some ticks and tick labels
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels, fontsize=18)
    ax2.set_yticks(times[::8][::5])
    ax2.set_yticklabels(
        ['{0:%Y-%m-%d}'.format(times[i]) for i in range(0, len(times), 8)][::5], 
        fontsize=14, rotation=0)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
    # increase tick width
    ax2.tick_params(width=1.5)
    ax2.legend(
        handles=[patch1, patch2], bbox_to_anchor=(0.95, 1.05), 
        ncol=2, frameon=False, fontsize=14)
    # Set some titles
    # plt.title('moist layer strength', loc='left', fontsize=10)
    # plt.title('Time Range: {0:%Y%m%d %HZ} - {1:%Y%m%d %HZ}'.format(times[0], times[-1]),
    #         loc='right', fontsize=10)
    return (fig, ax1, ax2)

def eml_char_ds_to_pgrid(eml_char_ds, pgrid, vertical_dim):
    gridded_eml_ds = (
        eml_char_ds.
        assign_coords({'pmean': eml_char_ds.pmean}).
        swap_dims({'eml_count': 'pmean'}).
        reindex(
            indexers={'pmean': pgrid}, 
            method='nearest', 
            tolerance=np.diff(list(pgrid) + [0])/2).
        swap_dims({'pmean': vertical_dim})
        )
    return gridded_eml_ds

def main():
    pass
         
if __name__ == '__main__':
    main()
