import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from typhon.physics import vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk
from typhon.plots import profile_p

import moist_layers as ml


def plot_eml_strength_map(eml_strength, lat, lon, time):
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
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_xticks(np.arange(int(lon.min()), int(lon.max()), 2)+1, crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(int(lat.min()), int(lat.max()), 1)+1, crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], 
    crs=ccrs.PlateCarree())
    ax.set_title(f"{str(time)[:19]}")
    cb = plt.colorbar(cs1, extend="max", orientation="vertical", shrink=1, pad=0.02)
    # cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
    cb.ax.set_ylabel(r"EML strength")
    cb.ax.tick_params(labelsize=10)

    return fig, ax
    
def main():
    eml_chars = ml.load_pickle('eml_chars.pickle')
    eml_ds = xr.concat(
        [eml_char.to_xr_dataset() for eml_char in eml_chars], 
        dim='eml_count')
    eml_ds = eml_ds.assign_coords(
        {'eml_count': np.arange(len(eml_ds.eml_count))})
    plot_eml_strength_map(eml_ds.strength, np.rad2deg(eml_ds.lat), np.rad2deg(eml_ds.lon), eml_ds.time[0].values)
    plt.savefig('/home/u/u300676/moist-layers/plots/eml_strength_map.png')


if __name__ == '__main__':
    main()