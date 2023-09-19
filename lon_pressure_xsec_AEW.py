from tkinter import E
import xarray as xr
import numpy as np
import argparse
import matplotlib.pyplot as plt
import glob
import global_land_mask as globe
from mpl_toolkits.axes_grid1 import make_axes_locatable

import eval_eml_chars as eec
from moist_layers import potential_temperature
from typhon.physics import (vmr2relative_humidity,
                           e_eq_mixed_mk, specific_humidity2vmr)
from metpy.calc import (equivalent_potential_temperature,
                        dewpoint_from_specific_humidity)
from metpy.units import units

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2021-08-04")
    args = parser.parse_args()
    eml_ds_paths = np.sort(
        np.concatenate(
            [glob.glob('/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                  'era5_3h_30N-S_eml_tropics_2021-07*.nc'), 
             glob.glob('/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                   'era5_3h_30N-S_eml_tropics_2021-08*.nc')]))
    # eml_ds_mean = xr.open_mfdataset(
    #     eml_ds_paths, concat_dim='time', combine='nested')
    eml_ds_mean_paths = np.sort(
        glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/geographical/'
                f'era5_3h_30N-S_eml_tropics_*_2021-07_geographical_mean.nc'
                ))

    eml_ds_mean = xr.open_mfdataset(
        eml_ds_mean_paths)
    eml_ds_mean = eml_ds_mean.sel(
            {
                'lat': 15,
                'lon': slice(-70, 20),
                'fulllevel': slice(43, eml_ds_mean.fulllevel.max()+1),
                # 'time': '2021-08-10'
            })
    dewpoint_mean = dewpoint_from_specific_humidity(
        eml_ds_mean.pfull_mean*units.Pa, eml_ds_mean.t_mean*units.K, eml_ds_mean.q_mean)
    theta_e_mean = equivalent_potential_temperature(
        eml_ds_mean.pfull_mean*units.Pa, eml_ds_mean.t_mean*units.K, dewpoint_mean)
    # theta_mean = potential_temperature(
    #         eml_ds_mean.ta, eml_ds_mean.pfull).values
    v_mean = eml_ds_mean.v_mean.values
    w_mean = eml_ds_mean.w_mean.values
    q_mean = eml_ds_mean.q_mean.values
    p_mean = eml_ds_mean.pfull_mean.mean('lon').values
    times = np.arange('2021-07-15', '2021-07-23', dtype='datetime64[D]')[::2]
    fig, axs = plt.subplots(
        ncols=3, nrows=len(times), figsize=(15, 5*len(times)), 
        sharey=True, sharex=True)
    for itime, time in enumerate(times):
        print(f'Plotting {time}', flush=True)
        eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested')
        eml_ds = eml_ds.sel(
            {
                'time': str(time),
                'lat': 15,
                'lon': slice(-70, 20),
                'fulllevel': slice(43, eml_ds.fulllevel.max()+1)
            }
        ).mean('time')
        eml_ds = eec.filter_gridded_eml_ds(
            eml_ds, 
            min_strength=0.2, 
            min_pmean=10000, 
            max_pmean=90000, 
            min_pwidth=5000, 
            max_pwidth=40000)
        
        # theta = potential_temperature(eml_ds.ta, eml_ds.pfull).values
        p = eml_ds.pfull.values
        t = eml_ds.t.values
        v = eml_ds.v.values
        q = eml_ds.q.values
        rh = eml_ds.rh.values
        q_ref = eml_ds.q_ref.values
        rh_ref = vmr2relative_humidity(
            specific_humidity2vmr(q_ref), p, t, e_eq=e_eq_mixed_mk)
        dewpoint = dewpoint_from_specific_humidity(
            p*units.Pa, t*units.K, eml_ds.q)
        theta_e = equivalent_potential_temperature(
            p*units.Pa, t*units.K, dewpoint)
        s1 = axs[itime, 1].contourf(
                eml_ds.lon, p_mean/100, theta_e-theta_e_mean, 
                levels=np.arange(-10, 11, 1),
                cmap='difference', extend='both')
        divider = make_axes_locatable(axs[itime, 1])
        cax = divider.append_axes("bottom", size="5%", pad=0.45)
        plt.axis('off')
        if itime == len(times)-1:
            plt.axis('on')
            cb1 = plt.colorbar(s1, orientation='horizontal', cax=cax)
            axs[itime, 1].set(xlabel='longitude')
            cb1.ax.set(xlabel='$\Theta_{e}$ anomaly / K')
        axs[itime, 1].set(ylim=[p_mean.max()/100, 100], ylabel='pressure / hPa',)
        s2 = axs[itime, 0].contourf(
                eml_ds.lon, p_mean/100, v-v_mean, 
                levels=np.arange(-15, 16, 1),
                cmap='difference', extend='both')
        divider = make_axes_locatable(axs[itime, 0])
        cax = divider.append_axes("bottom", size="5%", pad=0.45)
        cax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        plt.axis('off')
        if itime == len(times)-1:
            plt.axis('on')
            cb2 = plt.colorbar(s2, orientation='horizontal', cax=cax)
            cb2.ax.set(xlabel='v-wind anomaly / m s$^{-1}$')
            axs[itime, 0].set(xlabel='longitude')
        axs[itime, 0].set(title=str(time))
        s3 = axs[itime, 2].contourf(
                eml_ds.lon, p_mean/100, (q-q_mean)*1000, 
                levels=np.arange(-4, 4.5, 0.5),
                cmap='difference', extend='both')
        divider = make_axes_locatable(axs[itime, 2])
        cax = divider.append_axes("bottom", size="5%", pad=0.45)
        plt.axis('off')
        if itime == len(times)-1:
            plt.axis('on')
            cb3 = plt.colorbar(s3, orientation='horizontal', cax=cax)
            cb3.ax.set(xlabel='q anomaly / g kg$^{-1}$')
            axs[itime, 2].set(xlabel='longitude') 
        [axs[itime, i].contour(
            eml_ds.lon, 
            p_mean/100, 
            (rh - rh_ref)*100, 
            levels=np.arange(10, 45, 5),
            cmap='Greys') for i in range(3)]
    plt.subplots_adjust(hspace=0.002, wspace=0.2)
    plt.savefig('plots/era5/era5_aew_lon_pmean_xsec_thetae_v_q_long.png', dpi=300)

if __name__ == '__main__':
    main()