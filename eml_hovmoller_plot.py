import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import eval_eml_chars as eec
import argparse

def main():
    for year in range(2009, 2011):
        for lat in [0, 15]:
            # Load gridded EML data
            eml_ds_path = '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/' \
                        f'{year}/era5_3h_atlantic_{lat}N_{year}-07.nc'

            # eml_ds_paths = sorted(glob.glob(
            #     f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'
            #      'era5_3h_30N-S_eml_tropics_2020-07-*.nc'))
            # print('Loading dataset...', flush=True)
            # eml_ds = xr.open_mfdataset(
            #     eml_ds_paths, concat_dim='time', combine='nested').sel(
            #         {
            #             'lat': lat,
            #             'lon': slice(-60, 20),
            #         })
            eml_ds = xr.open_dataset(eml_ds_path).sortby('time')
            # eml_ds = eml_ds.sel(
            #     {
            #         'lat': lat, 
            #         'lon': slice(-60, 20),
            #         'time': slice('2021-07-01', '2021-08-01')
            #         })

            eml_ds = eec.filter_gridded_eml_ds(
                eml_ds.copy(),
                min_strength=0.3,
                min_pmean=50000,
                max_pmean=70000,
                min_pwidth=5000,
                max_pwidth=40000
                )
            # max_eml_ind_for_each_lon = np.full(eml_ds.lon.shape, np.nan)
            # eml_found_at_lon = np.any(~np.isnan(eml_ds.eml_strength), axis=1)
            # max_eml_ind = np.nanargmax(eml_ds.eml_strength[eml_found_at_lon, :], axis=1)
            # max_eml_ind_for_each_lon[eml_found_at_lon] = max_eml_ind
            # reduced_eml_ds = eml_ds.where(~np.isnan(eml_ds.eml_strength), drop=True)
            lons = eml_ds.lon.values
            times = eml_ds.time.values.astype('datetime64[h]').astype('O')
            # ua = eml_ds_v.ua.mean(dim='fulllevel').values
            va = eml_ds.v.where(
                ((eml_ds.pfull < 70000) & (eml_ds.pfull > 50000))).mean(
                    dim='fulllevel').values.squeeze()
            ua = eml_ds.u.where(
                ((eml_ds.pfull < 70000) & (eml_ds.pfull > 50000))).mean(
                    dim='fulllevel').values.squeeze()
            # rr = eml_ds_v.rain_rate.values
            # va_mean = va.mean(dim='time')
            # va_anom = (va - va_mean).values
            eml_strength = eml_ds.eml_strength.max(dim='fulllevel').values.squeeze() * 100
            eml_flag = ~np.isnan(eml_strength)
            eml_frac_northerly = np.logical_and(eml_flag, va < 0).sum() / (eml_flag).sum()
            eml_mean_ua = ua[eml_flag].mean()
            eml_mean_va = va[eml_flag].mean()
            print(f'EML mean u wind: {eml_mean_ua} m/s')
            print(f'EML mean v wind: {eml_mean_va} m/s')
            print(f'EML in northerly wind fraction: {eml_frac_northerly}')
            print('Creating Hovmoller plot...', flush=True)
            fig, ax1, ax2 = eec.hovmoller_plot(
                lons, times, data_fill=eml_strength, data_line=va,
                lat_min=lat, lat_max=lat, title='')
            ax2.set_title(f'EML fraction in northerly wind: {np.round(eml_frac_northerly, 2)}', loc='left')
            plt.tight_layout()
            plt.savefig(
                f'plots/revision/era5_hovmoller_atlantic_500-700hPa_{lat}deg_{year}-07_60W.png', 
                dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()