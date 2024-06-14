import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import eval_eml_chars as eec
import cartopy.crs as ccrs
from typhon.plots import worldmap
import global_land_mask as globe
from metpy.calc import static_stability
from metpy.units import units

from typhon.physics import density

def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))

def transform_to_log_length(u, v):
    arrow_lengths = np.sqrt(u*u + v*v)
    len_adjust_factor = np.log10(arrow_lengths + 1) / arrow_lengths
    return u*len_adjust_factor, v*len_adjust_factor

def main():
    month = '07'
    # Load gridded EML data
    eml_ds_paths = np.sort(
        glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/monthly_means/geographical/'
                f"era5_3h_30N-S_eml_tropics_*_2021-{month}_geographical_mean.nc"
                ))
    eml_ds_paths = [p for p in eml_ds_paths 
                    if (('_rh_' in p) or
                        ('_t_' in p) or
                        ('_u_' in p) or
                        ('_pfull_' in p) or
                        ('_v_' in p) or
                        ('_z_' in p) or
                        ('_w_' in p)
                        )
                    ]
    eml_ds = xr.open_mfdataset(
        eml_ds_paths)
    eml_ds = eml_ds.sel(
                {
                    'lon': slice(-20, 0),
                    'lat': slice(30, -30),
                    # 'fulllevel': slice(43, 91),
                }
            )
    # lat_mesh, lon_mesh = np.meshgrid(eml_ds.lat, eml_ds.lon) 
    # is_ocean = xr.DataArray(
    #             globe.is_ocean(lat_mesh, lon_mesh), 
    #             dims=['lon', 'lat'])
    # eml_ds = eml_ds.where(~is_ocean).mean('lon').dropna('lat', how='all')
    eml_ds = eml_ds.mean('lon').dropna('lat', how='all')

    # rho_mean = density(eml_ds.pfull_mean, eml_ds.t_mean)
    # rho_zonal_mean = rho_mean
    pfull_zonal_mean = eml_ds.pfull_mean
    # z_zonal_mean = eml_ds.z_mean
    v_zonal_mean = eml_ds.v_mean
    # w_zonal_mean = eml_ds.w_mean
    # w_zonal_mean = w_zonal_mean[1:, :] / np.diff(pfull_zonal_mean, axis=0) * np.diff(z_zonal_mean, axis=0)
    # psi_mean_x = rho_zonal_mean * v_zonal_mean * np.cos(np.deg2rad(eml_ds.lat))
    # psi_mean_y = rho_zonal_mean * w_zonal_mean * np.cos(np.deg2rad(eml_ds.lat))
    # psi_xy_angles = np.arctan2(psi_mean_x,psi_mean_y)*180.0/np.pi # calculate angles manually

    # psi_mean_x_log, psi_mean_y_log = transform_to_log_length(psi_mean_x, psi_mean_y)
    a = 6371 * 1e3
    psi_zonal_mean = np.zeros(v_zonal_mean.shape)
    for p_ind in range(1, 137):
            psi_zonal_mean[p_ind-1, :] = 2*np.pi*a*np.cos(np.deg2rad(eml_ds.lat)) / 9.81 * \
                    np.trapz(v_zonal_mean[:p_ind, :], pfull_zonal_mean[:p_ind, :], axis=0)
    fig, axs = plt.subplots(figsize=(11, 4))
    pc = axs.pcolormesh(
        eml_ds.lat, eml_ds.pfull_mean/100, eml_ds.rh_mean*100,
        cmap='Blues', vmin=0, vmax=100)
    cax1 = axs.inset_axes([1.01, 0, 0.015, 1])
    cb2 = plt.colorbar(
        pc, orientation="vertical", shrink=1, cax=cax1)
    cb2.ax.set(ylabel='RH / %')
    lat_2d = np.tile(
        eml_ds.lat, 
        len(eml_ds.fulllevel)).reshape(
            (len(eml_ds.fulllevel), len(eml_ds.lat)))

    psi_contours = np.arange(-30, 36, 6)

    CS = axs.contour(
        lat_2d, pfull_zonal_mean/100, psi_zonal_mean*1e-10, levels=psi_contours, colors='k', 
        negative_linestyles='dashed', linewidths=1)
    
    # axs.quiver(
    #     lat_2d[::2, ::5], eml_ds.pfull_mean[::2, ::5]/100, 
    #     psi_mean_x_log[::2, ::5], psi_mean_y_log[::2, ::5]*-1,
    #     scale=10, angles=psi_xy_angles[::2, ::5]
    #     )
    # c = axs.contour(
    #     lat_2d, eml_ds.pfull_mean/100, eml_ds.u_mean,
    #     cmap='RdYlGn', levels=np.arange(-12, 14, 2),
    #     vmin=-12, vmax=14,
    #     linewidths=1,
    # )
    # cax2 = axs.inset_axes([1.15, 0, 0.02, 1])
    # cb2 = plt.colorbar(
    #     c, orientation="vertical", shrink=1, pad=0.02, cax=cax2)
    # cb2.ax.set(ylabel='u-wind / ms$^{-1}$')
    # axs.vlines([0, 15], [100, 100], [1013.25, 1013.25], color='grey', ls='--')
    # axs.text(0, 80, 'Eq.', color='grey')
    # axs.text(15, 80, '15° N', color='grey')
    # axs.text(3, 700, 'AEJ', color='firebrick', weight='bold')
    axs.invert_yaxis()
    axs.set_ylim([1013.25, 100.00])
    axs.clabel(CS, inline=True, fontsize=8)
    # axs.clabel(c, inline=True, fontsize=8)
    axs.set_ylabel('Pressure / hPa', fontsize=12)
    axs.set_xlabel('Latitude / deg', fontsize=12)
    axs.contour(
        lat_2d,
        eml_ds.pfull_mean/100, 
        eml_ds.t_mean-273.15,
        levels=[0],
        linestyles='--', linewidths=1, colors='silver')
    axs.text(5, 550, '0° C', color='silver', weight='bold')
    # axs.text(5, 970, 'ML', color='yellowgreen', weight='bold')
    # ax2 = axs.twiny()
    # eml_ds_15N = eml_ds.sel(lat=15)
    # stability = static_stability(
    #     eml_ds_15N.pfull_mean*units.Pa, eml_ds_15N.t_mean*units.kelvin)
    # ax2.plot(stability*100, eml_ds_15N.pfull_mean/100, color='black')
    # ax2.invert_yaxis()
    # ax2.set_yscale('log')
    # ax2.set(
    #     ylim=[1013.25, 100.00], xlim=[0, 1e-3], 
    #     xlabel='static stability at 15° N / 10$^{-3}$ K hPa$^{-1}$')
    # ax2.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax2.set_xscale('log')
    plt.savefig(
         f'plots/paper/era5_african_circulation_0deg_psi_30S-30N_2021-{month}.png', dpi=300, 
         bbox_inches='tight')
if __name__ == '__main__':
    main()