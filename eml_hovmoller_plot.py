import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import eval_eml_chars as eec
import argparse

def main():
    # Load gridded EML data
    eml_ds_paths = np.sort(glob.glob(
        '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
        'era5_3h_30N-S_eml_tropics_2021-01-01*.nc'))

    eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested')
    lat = 15
    # eml_ds_mean = xr.open_mfdataset(
    #     eml_ds_paths, concat_dim='time', combine='nested').sel(
    #         {
    #             'lat': lat,
    #             'lon': slice(-180, -90),
    #         }
    #     ).mean('time')
    eml_ds = eml_ds.sel(
        {
            'lat': lat, 
            'lon': slice(-60, 20),
            'time': slice('2021-01-01', '2021-01-02')
            })#.mean('lat')

    eml_ds = eec.filter_gridded_eml_ds(
        eml_ds.copy(),
        min_strength=0.3,
        min_pmean=50000,
        max_pmean=70000,
        min_pwidth=5000,
        max_pwidth=40000)
    # max_eml_ind_for_each_lon = np.full(eml_ds.lon.shape, np.nan)
    # eml_found_at_lon = np.any(~np.isnan(eml_ds.eml_strength), axis=1)
    # max_eml_ind = np.nanargmax(eml_ds.eml_strength[eml_found_at_lon, :], axis=1)
    # max_eml_ind_for_each_lon[eml_found_at_lon] = max_eml_ind
    # reduced_eml_ds = eml_ds.where(~np.isnan(eml_ds.eml_strength), drop=True)
    lons = eml_ds.lon.values
    times = eml_ds.time.values.astype('datetime64[h]').astype('O')
    # ua = eml_ds_v.ua.mean(dim='fulllevel').values
    va = eml_ds.v.where(
        ((eml_ds.pfull < 75000) & (eml_ds.pfull > 65000))).mean(
            dim='fulllevel').values
    # rr = eml_ds_v.rain_rate.values
    # va_mean = va.mean(dim='time')
    # va_anom = (va - va_mean).values
    eml_strength = eml_ds.eml_strength.max(dim='fulllevel').values * 100
    fig, ax1, ax2 = eec.hovmoller_plot(
         lons, times, data_fill=eml_strength, data_line=va,
         lat_min=lat, lat_max=lat, title='Monsoon, 0.25°x0.25°')
    plt.tight_layout()
    plt.savefig(
        f'plots/paper/era5_hovmoller_cmap_atlantic_va_500-700hPa_{lat}deg_2021-01_60W.png', 
        dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()