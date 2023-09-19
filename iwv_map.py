import xarray as xr
import glob 
import numpy as np
import matplotlib.pyplot as plt
import global_land_mask as globe
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from typhon.plots import worldmap


eml_ds_paths = np.sort(
        np.concatenate(
            [glob.glob('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                  'gridded_monsoon_0p25deg_eml_tropics_2021-07*_iwv.nc'), 
             glob.glob('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                   'gridded_monsoon_0p25deg_eml_tropics_2021-08*_iwv.nc')]))
eml_ds = xr.open_mfdataset(
    eml_ds_paths, concat_dim='time', combine='nested')
eml_ds = eml_ds.sel(time=slice('2021-07-28', '2021-07-28'))

eml_ds_mean = xr.open_dataset(
    '/home/u/u300676/user_data/mprange/eml_data/gridded/'
    'monsoon_gridded_0p25deg_eml_tropics_mean_2021_07_08_iwv.nc' 
    )

# eml_ds_mean = eml_ds_mean.sel(lon=slice(-179.75, 179.75))

lat_mesh, lon_mesh = np.meshgrid(eml_ds_mean.lat, eml_ds_mean.lon) 
is_ocean = xr.DataArray(
        globe.is_ocean(lat_mesh, lon_mesh), 
        dims=['lon', 'lat'])
# eml_ds_mean = eml_ds_mean.where(is_ocean)
percentiles = np.arange(0, 102, 2)
iwv_bins = np.nanpercentile(eml_ds.iwv, percentiles)

iwv_p40 = iwv_bins[percentiles == 40][0]
iwv_p60 = iwv_bins[percentiles == 60][0]
iwv_p70 = iwv_bins[percentiles == 70][0]
iwv_p90 = iwv_bins[percentiles == 90][0]

iwv_p40_p60_ind = (~eml_ds.where(
    (eml_ds.iwv > iwv_p40) & 
    (eml_ds.iwv < iwv_p60)).iwv.isnull())
iwv_p40_p60_count = iwv_p40_p60_ind.sum('time').values
iwv_p40_p60_rh_mean = eml_ds.rh.where(iwv_p40_p60_ind).mean(['lat', 'lon', 'time']).values
iwv_p40_p60_rh_std = eml_ds.rh.where(iwv_p40_p60_ind).std(['lat', 'lon', 'time']).values
iwv_p40_p60_t_mean = eml_ds.ta.where(iwv_p40_p60_ind).mean(['lat', 'lon', 'time']).values
iwv_p40_p60_t_std = eml_ds.ta.where(iwv_p40_p60_ind).std(['lat', 'lon', 'time']).values
iwv_p40_p60_pfull_mean = eml_ds.pfull.where(iwv_p40_p60_ind).mean(['lat', 'lon', 'time']).values

iwv_p70_p90_ind = (~eml_ds.where(
    (eml_ds.iwv > iwv_p70) & 
    (eml_ds.iwv < iwv_p90)).iwv.isnull())
iwv_p70_p90_count = iwv_p70_p90_ind.sum('time').values
iwv_p70_p90_rh_mean = eml_ds.rh.where(iwv_p70_p90_ind).mean(['lat', 'lon', 'time']).values
iwv_p70_p90_rh_std = eml_ds.rh.where(iwv_p70_p90_ind).std(['lat', 'lon', 'time']).values
iwv_p70_p90_t_mean = eml_ds.ta.where(iwv_p70_p90_ind).mean(['lat', 'lon', 'time']).values
iwv_p70_p90_t_std = eml_ds.ta.where(iwv_p70_p90_ind).std(['lat', 'lon', 'time']).values
iwv_p70_p90_pfull_mean = eml_ds.pfull.where(iwv_p70_p90_ind).mean(['lat', 'lon', 'time']).values

fig = plt.figure(figsize=(8, 5))
# Use gridspec to help size elements of plot; small top plot and big bottom plot
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6], hspace=0.1)

ax1 = fig.add_subplot(
    gs[0, :], projection=ccrs.PlateCarree(central_longitude=0))
s = worldmap(
    eml_ds_mean.lat, eml_ds_mean.lon, eml_ds_mean.iwv, cmap='density',
    draw_coastlines=True, ax=ax1, fig=fig, vmin=0, vmax=60)
cbar2 = plt.colorbar(
        s, orientation='horizontal', pad=0.12, aspect=25, extendrect=True,
        ax=ax1)

cf1 = ax1.contourf(
    lon_mesh, lat_mesh, iwv_p40_p60_count.T, 
    levels=[4, 8],
    vmin=4, vmax=8,
    cmap='Greys', alpha=0.3,
    extend='neither', linewidths=0.1)

cf1 = ax1.contourf(
    lon_mesh, lat_mesh, iwv_p70_p90_count.T, 
    levels=[4, 8],
    vmin=4, vmax=8,
    cmap='Reds', alpha=0.3,
    extend='neither', linewidths=0.1)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(iwv_p40_p60_rh_mean*100, iwv_p40_p60_pfull_mean/100)
ax2.fill_betweenx(
   iwv_p40_p60_pfull_mean/100, 
   (iwv_p40_p60_rh_mean - iwv_p40_p60_rh_std)*100, 
   (iwv_p40_p60_rh_mean + iwv_p40_p60_rh_std)*100,
)
ax2.plot(iwv_p70_p90_rh_mean*100, iwv_p70_p90_pfull_mean/100)
ax2.fill_betweenx(
   iwv_p70_p90_pfull_mean/100, 
   (iwv_p70_p90_rh_mean - iwv_p70_p90_rh_std)*100, 
   (iwv_p70_p90_rh_mean + iwv_p70_p90_rh_std)*100,
   alpha=0.5
)
ax2.set(ylim=[1020, 100])

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(iwv_p40_p60_t_mean, iwv_p40_p60_pfull_mean/100)
ax3.fill_betweenx(
   iwv_p40_p60_pfull_mean/100, 
   (iwv_p40_p60_t_mean - iwv_p40_p60_t_std), 
   (iwv_p40_p60_t_mean + iwv_p40_p60_t_std),
   alpha=0.5
)
ax3.plot(iwv_p70_p90_t_mean, iwv_p70_p90_pfull_mean/100)
ax3.fill_betweenx(
   iwv_p70_p90_pfull_mean/100, 
   (iwv_p70_p90_t_mean - iwv_p70_p90_t_std), 
   (iwv_p70_p90_t_mean + iwv_p70_p90_t_std),
   alpha=0.5
)
ax3.set(ylim=[1020, 100])
# cf1 = ax1.contourf(
#     lon_mesh, lat_mesh, eml_ds_mean.iwv.T, 
#     levels=[iwv_p40, iwv_p60],
#     vmin=iwv_p40, vmax=iwv_p60,
#     colors='black', alpha=0.3,
#     extend='neither', linewidths=0.1)

# cf1 = ax1.contourf(
#     lon_mesh, lat_mesh, eml_ds_mean.iwv.T, 
#     levels=[iwv_p70, iwv_p90],
#     vmin=iwv_p70, vmax=iwv_p90,
#     colors='white', alpha=0.3,
#     extend='neither', linewidths=0.1)

plt.savefig(
    'plots/monsoon_gridded/monsoon_iwv_mean_map.png', dpi=300)