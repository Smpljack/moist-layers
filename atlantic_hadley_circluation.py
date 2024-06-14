import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typhon.physics import density
from typhon.plots import worldmap

for year in ['2022', '2016', '2015', '2014', '2013', '2012', '2011']:
    for month in ['01', '07']:
        eml_ds_paths = np.sort(
                glob.glob(
                        f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'
                        'monthly_means/geographical/'
                        f'era5_3h_30N-S_eml_tropics_*_{year}-{month}_geographical_mean.nc'
                        ))

        eml_ds = xr.open_mfdataset(eml_ds_paths)
        lat = eml_ds.lat
        pfull_mean = eml_ds.pfull_mean.T.load()
        rh_zonal_mean = eml_ds.rh_mean.T.mean('lon').load()
        rh_zonal_std = eml_ds.rh_std.T.mean('lon').load()
        neml_zonal_sum = eml_ds.n_eml_0p3.T.sum('lon').load()
        t_mean = eml_ds.t_mean.T.load()
        z_mean = eml_ds.z_mean.T.load()
        v_mean = eml_ds.v_mean.T.load()
        w_mean = eml_ds.w_mean.T.load()
        iwv_mean = eml_ds.iwv_mean.T.load()
        iwv_std = eml_ds.iwv_std.T.load()
        rho_mean = density(pfull_mean, t_mean)
        pfull_zonal_mean = pfull_mean.mean('lon')
        z_zonal_mean = z_mean.mean('lon')
        rho_zonal_mean = rho_mean.mean('lon')
        v_zonal_mean = v_mean.mean('lon')
        w_zonal_mean = w_mean.mean('lon')
        iwv_zonal_mean = iwv_mean.mean('lon')
        iwv_zonal_std = iwv_std.mean('lon')
        w_zonal_mean = w_zonal_mean[:, 1:] / np.diff(pfull_zonal_mean) * np.diff(z_zonal_mean)
        psi_mean_x = rho_zonal_mean * v_zonal_mean * np.cos(np.deg2rad(lat))
        psi_mean_y = rho_zonal_mean * w_zonal_mean * np.cos(np.deg2rad(lat))
        mass_flux_zonal_mean = np.sqrt(psi_mean_x**2 + psi_mean_y**2)
        #p-grid for mass flux, defined on intermediate pressure levels
        pfull_mass_flux_zonal_mean = (pfull_zonal_mean[:, :-1].values + pfull_zonal_mean[:, 1:].values)/2
        psi_zonal_mean = np.zeros(v_zonal_mean.shape)



        eml_ds_paths_ms = np.sort(
                glob.glob(
                        f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{year}/'
                        'monthly_means/moisture_space/'
                        f'era5_3h_30N-S_eml_tropics_atlantic_*_{year}-{month}_moisture_space_mean.nc'
                        ))
        eml_ds_paths_ms = [p for p in eml_ds_paths_ms if 
                        '_pfull_' in p or
                        '_rh_' in p or
                        '_w_' in p or
                        '_n_eml_' in p or
                        '_t_' in p
                        ]

        eml_ds_ms = xr.open_mfdataset(eml_ds_paths_ms)
        pfull_mean_grouped = eml_ds_ms.pfull_mean.T.load()
        rh_mean_grouped = eml_ds_ms.rh_mean.T.load()
        w_mean_grouped = eml_ds_ms.w_mean.T.load()
        n_eml_grouped = eml_ds_ms.n_eml_0p3.T.load()
        t_mean_grouped = eml_ds_ms.t_mean.T.load()
        percentiles_2d = np.tile(
            np.arange(1, 100, 2), 
            len(eml_ds_ms.fulllevel)).reshape((len(eml_ds_ms.fulllevel), 50)).T
        psi = np.zeros(w_mean_grouped.shape)
        # moisture space stream function
        for i_iwv, pcol in enumerate(pfull_mean_grouped[1:, :]):
            i_iwv += 1
            for i_lev, plev in enumerate(pcol):
                psi[i_iwv, i_lev] = psi[i_iwv-1, i_lev] + 0.02/9.81 * w_mean_grouped[i_iwv, i_lev]
        # zonal mean stream function
        a = 6371 * 1e3
        for p_ind in range(1, len(pfull_zonal_mean[0, :])+1):
                psi_zonal_mean[:, p_ind-1] = 2*np.pi*a*np.cos(np.deg2rad(lat)) / 9.81 * \
                        np.trapz(v_zonal_mean[:, :p_ind], pfull_zonal_mean[:, :p_ind], axis=1)
        lat2d = np.meshgrid(lat, pfull_zonal_mean[0, :])[0].T

        fig = plt.figure(figsize=(7, 14))
        gs = gridspec.GridSpec(
                nrows=3, ncols=1, height_ratios=[0.5, 0.1, 1], figure=fig, hspace=0.25,
        )
        gs0 = gridspec.GridSpecFromSubplotSpec(
                nrows=1, ncols=1, subplot_spec=gs[0])
        gs1 = gridspec.GridSpecFromSubplotSpec(
                nrows=1, ncols=1, hspace=0.15, 
                subplot_spec=gs[1])
        gs2 = gridspec.GridSpecFromSubplotSpec(
                nrows=2, ncols=1, hspace=0.15, height_ratios=[1, 1],
                subplot_spec=gs[2])
        ax1 = fig.add_subplot(gs0[0])
        c1 = ax1.pcolormesh(
            percentiles_2d, 
            pfull_mean_grouped/100, 
            n_eml_grouped,
        #     levels=np.arange(0, 100, 10),
        #     vmin=0, vmax=100,
            cmap='density', alpha=1,
        #     extend='max'
            )
        # c1 = ax1.contourf(
        #     percentiles_2d[:, 42:], 
        #     pfull_mean_grouped/100, 
        #     rh_mean_grouped*100,
        #     levels=np.arange(0, 100, 10),
        #     vmin=0, vmax=100,
        #     cmap='Blues', alpha=1,
        #     extend='max')
        cb = plt.colorbar(c1, ax=ax1)
        cb.ax.set(ylabel='number of EMLs / -')
        ax1.invert_yaxis()
        ax1.set(xlabel='IWV percentile / -', ylabel='pressure / hPa')
        ax1.contour(
            percentiles_2d, 
            pfull_mean_grouped/100, 
            t_mean_grouped,
            levels=[273.15],
            colors='white',
            linestyles='--'
            )
        CS = ax1.contour(
            percentiles_2d, 
            pfull_mean_grouped/100, 
            psi*1e3,
            levels=np.arange(0, 3.3, 0.3),
            vmin=0, vmax=3,
            colors='k', negative_linestyles='dashed',
            extend='max'
            )
        ax1.clabel(CS, inline=True, fontsize=8)

        ax2 = fig.add_subplot(gs1[0])
        ax2.plot(lat, iwv_zonal_mean, color='black')
        # ax2.fill_between(lat, iwv_zonal_mean-iwv_zonal_std, iwv_zonal_mean+iwv_zonal_std, facecolor='black', alpha=0.5)
        ax2.set(xlabel='latitude / deg', ylabel='IWV / kg m$^{-2}$', ylim=[10, 60], xlim=[-30, 30])
        cb = plt.colorbar(c1, ax=ax2)
        cb.ax.set(ylabel='nothing')
        plt.tick_params('x', labelbottom=True)

        if month == '01':
            psi_contours = np.arange(0, 45, 5)
        elif month == '07':
            psi_contours = np.arange(-40, 5, 5)

        ax3 = fig.add_subplot(gs2[0])
        CF = ax3.contourf(
                lat2d, pfull_zonal_mean/100, rh_zonal_mean*100, 
                levels=np.arange(0, 110, 10),
                cmap='Blues', vmin=0, vmax=100, )
        cb = plt.colorbar(CF, ax=ax3)
        cb.ax.set_ylabel('RH / %')
        CS = ax3.contour(
                lat2d, pfull_zonal_mean/100, psi_zonal_mean*1e-10, levels=psi_contours, colors='k', 
                negative_linestyles='dashed')
        ax3.clabel(CS, inline=True, fontsize=8)
        ax3.invert_yaxis()
        ax3.set(ylabel='pressure / hPa', xlim=[-30, 30])
        ax3.invert_yaxis()
        plt.tick_params('x', labelbottom=False)

        ax4 = fig.add_subplot(gs2[1])
        pc = ax4.pcolormesh(
                lat2d, pfull_zonal_mean/100, neml_zonal_sum,
                cmap='density', vmin=0, vmax=16000)
        cb = plt.colorbar(pc, ax=ax4)
        cb.ax.set_ylabel('number of EMLs / -')
        # cb.ax.set_ylim([0, 100])

        CS = ax4.contour(    
                lat2d, pfull_zonal_mean/100, psi_zonal_mean*1e-10, levels=psi_contours, colors='k', 
                negative_linestyles='dashed')
        ax4.clabel(CS, inline=True, fontsize=8)
        # ax.quiver(lat2d[::2, :-1], pfull_zonal_mean[::2, :-1], psi_mean_x[::2, :-1], psi_mean_y[::2, :])
        ax4.set(xlabel='latitude / deg', ylabel='pressure / hPa', xlim=[-30, 30])
        plt.tick_params('x', labelbottom=True)



        [ax.set_ylim([1000, 50]) for ax in [ax1, ax3, ax4]]
        plt.savefig(f'plots/revision/hadley_cell_atlantic_moisture_space_n_eml_{year}-{month}.png', dpi=300, bbox_inches='tight')