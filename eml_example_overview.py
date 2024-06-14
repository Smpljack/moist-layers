from cProfile import label
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from typhon.physics import vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk
from typhon.plots import label_axes
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from decimal import Decimal
import argparse
import glob
import matplotlib.gridspec as gridspec
from metpy.calc import divergence, potential_temperature
from metpy.units import units

import moist_layers as ml
import eval_eml_chars as eec
from typhon.plots import worldmap
sns.set_context('paper')

parser = argparse.ArgumentParser()
parser.add_argument("--time", type=str,
                help="timestamp",
                default="2021-07-19T00:00:00")
args = parser.parse_args()
eml_ds_paths = glob.glob(
    '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
    f'era5_3h_30N-S_eml_tropics_{args.time}.nc')
eml_ds_mean_paths = np.sort(
        glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/geographical/'
                f'era5_3h_30N-S_eml_tropics_*_2021-07_geographical_mean.nc'
                ))  
eml_ds_mean_paths = [p for p in eml_ds_mean_paths
                           if ('atlantic' not in p) 
                           and ('west_pacific' not in p)
                           and ('east_pacific' not in p)]
eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested').sel(
            time=args.time
        )
eml_ds = eml_ds.assign(
    {'theta': potential_temperature(
        eml_ds.pfull*units.pascal, eml_ds.t*units.kelvin)})
eml_ds_mean = xr.open_mfdataset(eml_ds_mean_paths)
eml_ds = eml_ds.sel(
                {'lon': slice(-60, 0), 'lat': slice(30, 0)}
            )
eml_ds_xsec = eml_ds.sel({'lon': -45})
eml_ds_point = eml_ds_xsec.sel({'lat': 15})

eml_ds_mean = eml_ds_mean.sel(
                {'lon': slice(-60, 0), 'lat': slice(30, 0)}
            ).mean(['lat', 'lon'])
isentrope = 316
sns.set_context('paper')
# fig, ax = plt.subplots(ncols=3, figsize=(10,5), sharey=False)
fig = plt.figure(figsize=(9,7))
gs = gridspec.GridSpec(
    nrows=2, ncols=3, height_ratios=[1, 1], width_ratios=[2, 1, 1], hspace=0.1,)
ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
theta_isentrope_bool = (np.abs(eml_ds.theta - isentrope*units.kelvin) == 
    np.abs(eml_ds.theta - isentrope*units.kelvin).min(dim='fulllevel'))
q_isentrope = eml_ds.q.where(theta_isentrope_bool).max(dim='fulllevel')
u_isentrope = eml_ds.u.where(theta_isentrope_bool).max(dim='fulllevel') 
v_isentrope = eml_ds.v.where(theta_isentrope_bool).max(dim='fulllevel') 
cs1 = ax1.pcolormesh(
        eml_ds.lon,
        eml_ds.lat,
        q_isentrope,
        vmin=0,
        vmax=0.01,
        cmap="density",
        transform=ccrs.PlateCarree(),
    )
q = ax1.quiver(
    eml_ds.lon[::10], eml_ds.lat[::10], 
    u_isentrope[::10, ::10], 
    v_isentrope[::10, ::10],
    alpha=0.5)
ax1.plot(
    [eml_ds_xsec.lon, eml_ds_xsec.lon], 
    [eml_ds_xsec.lat.min(), eml_ds_xsec.lat.max()],
    color='black', linestyle='-',
    transform=ccrs.PlateCarree(),
    )
ax1.scatter(
    eml_ds_point.lon, eml_ds_point.lat,
    color=sns.color_palette('colorblind')[2], marker='x', s=100)
cb1 = plt.colorbar(
    cs1, extend="max", orientation="vertical", shrink=0.9, pad=0.03)
cb1.ax.set_ylabel(
    r"q$_{\Theta316}$ "
    r"/ kg kg$^{-1}$", fontsize=12)
cb1.ax.tick_params(labelsize=10)
ax1.coastlines(resolution="10m", linewidth=0.6)
ax1.set_extent([-60, -0, 0, 30], crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(0, 31, 10), crs=ccrs.PlateCarree())
lat_formatter = LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.set_xticks(np.arange(-60, 10, 10), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.tick_params(labelsize=12)
ax1.set_xlabel(None)
ax1.set_ylabel(None)
ax1.set_aspect(0.8)
eml_ds_filtered = eec.filter_gridded_eml_ds(
        eml_ds.copy(), 
        min_strength=0.3, 
        min_pmean=50000, 
        max_pmean=70000, 
        min_pwidth=5000, 
        max_pwidth=40000,
        )
heights = np.arange(2000, 6000, 50)
eml_labels3d = eec.get_3d_height_eml_labels_griddata(
    eml_ds_filtered, eml_ds_filtered.lat, eml_ds_filtered.lon, heights=heights,
    height_tolerance=50)
eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
lon_mesh, lat_mesh = np.meshgrid(eml_ds_filtered.lon, eml_ds_filtered.lat)
ax1.contourf(
    eml_ds_filtered.lon, eml_ds_filtered.lat, 
    xr.where(eml_labels2d != 0, 1, 0), 
    levels=[0.5, 1],
    colors=['black', 'white'], alpha=0.1)
# ax4 = fig.add_subplot(gs[0, 1]) 
# ax4.plot(
#     eml_ds_mean.q_mean, 
#     eml_ds_mean.pfull_mean/100, 
#     color='black', label='July 2021 mean')
# ax4.plot(eml_ds_point.q, eml_ds_point.pfull/100, 
#         label=r'$1°\times1°$ mean at X', color=sns.color_palette('colorblind')[2])
# ax4.plot(eml_ds_point.q_ref, eml_ds_point.pfull/100, 
#                 label=r'ref profile at X', color=sns.color_palette('colorblind')[3])
# ax4.invert_yaxis()
# ax4.set_xscale('log')
# ax4.set_yticklabels([])
# ax4.set_xlim([1e-7, 0.03])
# ax4.set_ylim([1013.25, 100.00])
# ax4.set_xlabel('q / kg kg$^{-1}$', fontsize=12)
# ax4.set_ylabel('Pressure / hPa', fontsize=12)
# ax4.legend(loc='upper right')

ax2 = fig.add_subplot(gs[1, 0])
p = ax2.pcolormesh(
    eml_ds_xsec.lat, eml_ds_xsec.pfull/100,
    (eml_ds_xsec.rh - eml_ds_mean.rh_mean)*100,
    cmap='PuOr', vmin=-50, vmax=50)
ax2.plot([eml_ds_point.lat, eml_ds_point.lat], [1013.25, 100],
            color=sns.color_palette('colorblind')[2], linestyle='-',
            )
cb2 = plt.colorbar(p, extend="max", orientation="vertical", shrink=1, pad=0.02)
# cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
cb2.ax.set_ylabel(r"RH-$\overline{\mathrm{RH}}$ / %")
ax2.set_xticks(np.arange(0, 31, 10), crs=ccrs.PlateCarree())
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lat_formatter)
cb2.ax.tick_params(labelsize=10)
ax2.invert_yaxis()
ax2.set_ylim([1013.25, 100.00])
ax2.set_ylabel('Pressure / hPa', fontsize=12)
# ax2.set_xlabel('Latitude', fontsize=12)
ax2.tick_params(labelsize=12)

lat = np.tile(eml_ds_xsec.lat, (137, 1))
c = ax2.contour(
    lat, eml_ds_xsec.pfull/100, eml_ds_xsec.theta,
    levels=[isentrope],
    colors='black', label=f'{isentrope} K isentrope',
    shrink=0.9
    )
ax2.clabel(
    c, c.levels, inline=True, fontsize=10, 
    fmt={isentrope: fr'$\Theta${isentrope}'}, manual=[(25, 650)])
# c = ax2.contour(
#     lat, eml_ds_xsec.pfull/100, eml_ds_xsec.theta,
#     levels=np.arange(300, 365, 5),
#     colors='gray', alpha=0.5
#     )

# plt.colorbar(c, )
# div = divergence(eml_ds.u, eml_ds.v)
# div_xsec = div.sel({'lon': eml_ds_xsec.lon})
# lat = np.tile(eml_ds_xsec.lat, (137, 1))
# ax2.contour(
#     lat, eml_ds_xsec.pfull, eml_ds_xsec.w/100*3600, #hPa h-1 
#     levels=[-5, -3, -2, -1, 1, 2, 3, 5],
#     cmap='bwr_r')
# w = eml_ds_xsec.w[1:, :] * (
#     np.diff(eml_ds_xsec.z, axis=0) / 
#     np.diff(eml_ds_xsec.pfull, axis=0))

# q = ax2.quiver(
#     lat[1:, ::10], eml_ds_xsec.pfull[1:, ::10], 
#     eml_ds_xsec.v[1:, ::10], w[:, ::10])
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(
    eml_ds_mean.rh_mean*100, eml_ds_mean.pfull_mean/100, 
    color='black', label='July 2021 mean')
ax3.plot(eml_ds_point.rh*100, eml_ds_point.pfull/100, 
        label=r'profile at X', color=sns.color_palette('colorblind')[2])
ref_rh_point = vmr2relative_humidity(
    specific_humidity2vmr(eml_ds_point.q_ref), 
    eml_ds_point.pfull,
    eml_ds_point.t, e_eq=e_eq_mixed_mk)
ax3.plot(ref_rh_point*100, eml_ds_point.pfull/100, ls='--',
         label=r'ref. profile at X', color=sns.color_palette('colorblind')[2])
ax3.invert_yaxis()
ax3.set_yticklabels([])
ax3.set_xlim([0, 100])
ax3.set_ylim([1013.25, 100.00])
ax3.set_xlabel('RH / %', fontsize=12)
ax3.legend(fontsize=7, loc='best', bbox_to_anchor=(0.4, 0.6))

ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(
    np.diff(eml_ds_point.v)/np.diff(eml_ds_point.z), 
    eml_ds_point.pfull[1:]/100, 
    label='v shear', color=sns.color_palette('colorblind')[2])
ax4.plot(
    np.diff(eml_ds_point.u)/np.diff(eml_ds_point.z), 
    eml_ds_point.pfull[1:]/100, ls='--',
    label='u shear', color=sns.color_palette('colorblind')[2])
ax4.invert_yaxis()
ax4.set_yticklabels([])
# ax4.set_xlim([0, 100])
ax4.set_ylim([1013.25, 100.00])
ax4.set_xlabel('wind shear / 1/s', fontsize=12)
ax4.legend(fontsize=7, loc='best', bbox_to_anchor=(0.4, 0.6))

vmr_ref = specific_humidity2vmr(eml_ds_point.q_ref.values)
eml_chars = ml.eml_characteristics(
                    specific_humidity2vmr(eml_ds_point.q.values)[::-1], 
                    vmr_ref[::-1], 
                    eml_ds_point.t.values[::-1], 
                    eml_ds_point.pfull.values[::-1], 
                    eml_ds_point.z.values[::-1],
                    min_eml_p_width=5000.,
                    min_eml_strength=0.2,
                    p_min=20000.,
                    p_max=90000.,
                    z_in_km=False,
                    p_in_hPa=False,
                    lat=eml_ds_point.lat.values,
                    lon=eml_ds_point.lon.values,
                    time=eml_ds_point.time.values,)
                    
if ~np.isnan(eml_chars.strength)[0]:
    eml_ds = eml_chars.to_xr_dataset()
    eml_ds_filtered = eec.filter_eml_data(
        eml_ds, 
        min_strength=0.3, 
        min_pmean=20000, 
        max_pmean=90000, 
        min_pwidth=5000, 
        max_pwidth=40000,
        )
    if len(eml_ds_filtered.eml_count) > 0:
        strongest = eml_ds_filtered.strength.values.argmax()
        # ax3.text(50, 400, 'EML strength:\n'
        #                 f'{np.round(eml_ds_filtered.strength[strongest].values*100, 2)} %', 
        #         fontsize=10)
        pmean_ind = abs(eml_ds_point.pfull[::-1] - eml_ds_filtered.pmean[strongest].values).argmin()
        pmin_ind = abs(eml_ds_point.pfull[::-1] - eml_ds_filtered.pmin[strongest].values).argmin()
        pmax_ind = abs(eml_ds_point.pfull[::-1] - eml_ds_filtered.pmax[strongest].values).argmin()
        for pmin, pmax in zip(eml_ds_filtered.pmin, eml_ds_filtered.pmax):
            p_ind = eml_ds_point.pfull >= pmin
            p_ind &= eml_ds_point.pfull <= pmax
            pfill = eml_ds_point.pfull[p_ind].values
            ax3.fill_betweenx(
                pfill/100, 
                ref_rh_point[p_ind]*100, 
                eml_ds_point.rh[p_ind]*100, color=sns.color_palette('colorblind')[0], alpha=0.5,
                         label='EML')
#         # ax3.hlines(eml_ds_filtered.pmean[strongest]/100, 
#         #             eml_ds_point.rh[::-1][pmean_ind.values]*100-10, eml_ds_point.rh[::-1][pmean_ind.values]*100+10,
#         #             color=sns.color_palette('colorblind')[3])
#         # ax3.hlines(eml_ds_filtered.pmin[strongest]/100, 
#         #             eml_ds_point.rh[::-1][pmin_ind.values]*100-5, eml_ds_point.rh[::-1][pmin_ind.values]*100+5,
#         #             color=sns.color_palette('colorblind')[3])
#         # ax3.hlines(eml_ds_filtered.pmax[strongest]/100, 
#         #             eml_ds_point.rh[::-1][pmax_ind.values]*100-5, eml_ds_point.rh[::-1][pmax_ind.values]*100+5,
#         #             color=sns.color_palette('colorblind')[3]) 
#     elif (len(eml_ds.eml_count) > 0):
#         strongest = eml_ds.strength.values.argmax()
#         if (eml_ds.strength[strongest] < 0.3):
#             ax3.text(50, 400, 'EML not\nstrong enough:\n'
#                             f'{np.round(eml_ds.strength[strongest].values*100, 2)} %')
#         elif eml_ds.pmean[strongest] < 20000:
#             ax3.text(50, 400, 'EML pmean\ntoo low:\n'
#                             f'{np.round(eml_ds.pmean[strongest].values/100, 2)} hPa')
#         elif eml_ds.pmean[strongest] > 90000:
#             ax3.text(50, 400, 'EML pmean\ntoo high:\n'
#                             f'{np.round(eml_ds.pmean[strongest].values/100, 2)} hPa')
#         elif eml_ds.pwidth[strongest] < 5000:
#             ax3.text(50, 400, 'EML thickness\ntoo low:\n'
#                             f'{np.round(eml_ds.pwidth[strongest].values/100, 2)} hPa')
#         elif eml_ds.pwidth[strongest] > 40000:
#             ax3.text(50, 400, 'EML thickness\ntoo high:\n'
#                             f'{np.round(eml_ds.pwidth[strongest].values/100, 2)} hPa')
# else: 
#     ax3.text(70, 400, 'No EML') 
# ax3.tick_params(labelsize=12)

# fig.suptitle(f'ERA5, {args.time}')
label_axes([ax1, ax2, ax3], labels=['a)', 'b)', 'c)'], fontsize=12)
plt.savefig(f'plots/paper/era5_eml_example_isentrope_shear_{args.time}.png', dpi=300)

# eec.make_movie('plots/eml_monsoon_overview_rh_eml_def_*.pn    g',
#                'videos/eml_example_overview_movie_rh_eml_def.mp4')