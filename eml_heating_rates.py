import xarray as xr
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main():
    ds = xr.open_dataset(
        '/home/u/u300676/user_data/mprange/eml_data/gridded/'
        'hr_gridded_monsoon_0p25deg_eml_tropics_2021-07-28T00.nc')
    p = ds.pfull.mean(['lat', 'lon'])
    t = ds.ta.mean(['lat', 'lon']) 
    q = ds.hus.mean(['lat', 'lon'])
    rh = ds.rh.mean(['lat', 'lon']) * 100
    Q = ds.heating_rate_lw.mean(['lat', 'lon'])
    fig, axs = plt.subplots(ncols=3, sharey=True)
    axs[0].plot(t, p, color='black', lw=1.5)
    axs[1].plot(rh, p, color='black', lw=1.5)
    axs[2].plot(Q, p, color='black', lw=1.5)
    axs[0].set_ylim([102000, 5000])
    for lon in range(-170, 0, 10):
        ds_col = ds.sel({'lon': lon, 'lat': 0})
        p = ds_col.pfull.values
        t = ds_col.ta.values
        q = ds_col.hus.values
        rh = ds_col.rh.values * 100
        Q = ds_col.heating_rate_lw.values
        axs[0].plot(t, p, lw=0.5, alpha=0.5,)
        axs[1].plot(rh, p, lw=0.5, alpha=0.5,)
        axs[2].plot(Q, p, lw=0.5, alpha=0.5,)
        axs[0].set_ylim([102000, 5000])
        axs[0].set(ylabel='pressure / hPa', xlabel='temperature / K')
        axs[1].set(xlabel='relative humidity / %')
        axs[2].set(xlabel='heating rate /\nK day$^{-1}$')
    plt.savefig(
        'test_plots/heating_rates/mean_profile.png', 
        dpi=300, bbox_inches='tight')
if __name__ == '__main__':
    main()