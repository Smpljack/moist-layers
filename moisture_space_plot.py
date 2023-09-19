import xarray as xr
import glob 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typhon.plots import worldmap
import cmasher as cmr
from metpy.calc import moist_static_energy
from metpy.units import units

eml_ds_paths = np.sort(
        glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/moisture_space/'
                'era5_3h_30N-S_eml_tropics_atlantic_*_2021-01_moisture_space_mean.nc'
                ))

eml_ds = xr.open_mfdataset(eml_ds_paths)
# lat_mesh, lon_mesh = np.meshgrid(eml_ds.lat, eml_ds.lon)
# is_ocean = xr.DataArray(
#         globe.is_ocean(lat_mesh, lon_mesh), 
#         dims=['lon', 'lat'])
# eml_ds = eml_ds.where(is_ocean)

# crh_bins = np.arange(0, 1.05, 0.05)
# percentiles = np.arange(0, 102, 2)
# iwv_bins = np.nanpercentile(eml_ds.iwv, percentiles)
# eml_ds_grouped = eml_ds.groupby_bins(
#     'iwv', iwv_bins)
# eml_ds_iwv_grouped_mean = eml_ds_grouped.mean()
# eml_ds_iwv_grouped_std = eml_ds_grouped.std()

pfull_mean_grouped = eml_ds.pfull_mean.T[:, 42:].load()
z_mean_grouped = eml_ds.z_mean.T[:, 42:].load()
# zhalf_mean_grouped = eml_ds.zhalf_mean.T[:, 42:].load()
# phalf_mean_grouped = np.zeros(zhalf_mean_grouped.shape)
# for bin in range(50):
#     pfull_interp = interp1d(
#         z_mean_grouped[bin, :], pfull_mean_grouped[bin, :],
#         fill_value='extrapolate')
#     phalf_mean_grouped[bin, :] = pfull_interp(zhalf_mean_grouped[bin, :])
rh_mean_grouped = eml_ds.rh_mean.T[:, 42:].load()
rh_std_grouped = eml_ds.rh_std.T[:, 42:].load()
w_mean_grouped = eml_ds.w_mean.T[:, 42:].load()
q_mean_grouped = eml_ds.q_mean.T[:, 42:].load()
t_mean_grouped = eml_ds.t_mean.T[:, 42:].load()
mse_mean_grouped = moist_static_energy(
    z_mean_grouped*units.meter, 
    t_mean_grouped*units.kelvin,
    q_mean_grouped)
# w_mean_grouped = wa_mean_grouped[:, 1:] * \
# w_mean_grouped = wa_mean_grouped[:, 1:] * \
#                  np.diff(phalf_mean_grouped) / \
#                      np.diff(zhalf_mean_grouped) # pressure velocity
percentiles_2d = np.tile(
    np.arange(1, 100, 2), 
    len(eml_ds.fulllevel)).reshape((len(eml_ds.fulllevel), 50)).T

psi = np.zeros(w_mean_grouped.shape)
for i_iwv, pcol in enumerate(pfull_mean_grouped[1:, :]):
    i_iwv += 1
    for i_lev, plev in enumerate(pcol):
        psi[i_iwv, i_lev] = psi[i_iwv-1, i_lev] + 0.02/9.81 * w_mean_grouped[i_iwv, i_lev]
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
cf1 = axs[0].contourf(
    percentiles_2d[:, 42:], 
    pfull_mean_grouped/100, 
    mse_mean_grouped,
    levels=np.arange(300, 350, 5),
    vmin=300, vmax=350,
    cmap='density')
axs[0].set(
    ylim=[0, 17], xlim=[0, 100], ylabel='pressure / hPa', 
    xlabel='Percentile of IWV')
# cbar1 = plt.colorbar(
#         cf1, orientation='horizontal', pad=0.12, aspect=25, extendrect=True,
#         ax=axs[0])
# cbar1.ax.set(xlabel=r'$\overline{\mathrm{RH}}$ / %')
# c1 = axs[0].contour(
#     percentiles_2d[:, 42:], 
#     pfull_mean_grouped/100, 
#     rh_std_grouped*100,
#     levels=np.arange(0, 32, 2),
#     vmin=0, vmax=30,
#     cmap='Greys', alpha=0.3,
#     extend='max')
cmap = cmr.get_sub_cmap('RdYlGn_r', 0.5, 1)
c1 = axs[0].contour(
    percentiles_2d[:, 42:], 
    pfull_mean_grouped/100, 
    psi*1000,
    levels=np.arange(0, 4.2, 0.3),
    vmin=0, vmax=3.9,
    cmap=cmap, alpha=1,
    extend='max')
cbar1 = plt.colorbar(
        cf1, orientation='horizontal', pad=0.12, aspect=25, extendrect=False,
        ax=axs[0])
cbar1.ax.set(
    xlabel=r'$\overline{MSE}$ / kJ kg$^{-1}$')
cf2 = axs[1].contourf(
    percentiles_2d[:, 42:], 
    pfull_mean_grouped/100, 
    rh_std_grouped*100,
    levels=np.arange(0, 32, 2),
    vmin=0, vmax=30,
    cmap='Blues',
    extend='max')
# c1 = axs[1].contour(
#     percentiles_2d[:, 42:], 
#     pfull_mean_grouped/100, 
#     w_mean_grouped*3600/100,
#     levels=np.arange(-5, 5.3, 0.3),
#     vmin=-2, vmax=2,
#     cmap='RdYlGn', alpha=0.5,
#     extend='max')
axs[1].set(ylim=[1020, 50], xlim=[0, 100], xlabel='Percentile of IWV')
cbar2 = plt.colorbar(
        cf2, orientation='horizontal', pad=0.12, aspect=25, extendrect=True,
        ax=axs[1])
cbar2.ax.set(xlabel=r'$\sigma$(RH) / %')
plt.savefig(
    'plots/era5/era5_moisture_space_mse_mean_psi_p_iwv_2021-01_atlantic.png', 
    dpi=300)