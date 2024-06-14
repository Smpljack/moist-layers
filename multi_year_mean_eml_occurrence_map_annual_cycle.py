from turtle import filling
import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.axes as maxes
import eval_eml_chars as eec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from typhon.plots import worldmap
from typhon.plots import label_axes
import global_land_mask as globe
import seaborn as sns

from moist_layer_correlations import get_convective_array, iorg
from eml_occurence_map_annual_cycle import plot_monthly_eml_occurence

years = [str(y) for y in range(2009, 2024)]
base_path = '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
fig = plt.figure(
        figsize=(10, 5))
gs = gridspec.GridSpec(
    nrows=4, ncols=2, width_ratios=[3, 1], hspace=0.001, wspace=0.1)
axs = []
for imonth, month in enumerate(['01', '04', '07', '10']):
    ax = fig.add_subplot(
        gs[imonth, 0], projection=ccrs.PlateCarree(central_longitude=0))
    # Load gridded EML data
    eml_ds_paths = []
    eml_ds_paths = np.sort(np.concatenate(
        [[p for p in glob.glob(
                base_path + f'{year}/monthly_means/geographical/'
                f'era5_3h_30N-S_eml_tropics_*_{year}-{month}_geographical_mean.nc'
                )] for year in years], axis=0))
    eml_ds_paths = [
        p for p in eml_ds_paths 
        if ('_pfull_mean_' in p) 
        or ('_n_eml_' in p)
        or ('_rain_rate_' in p)
        or ('_iwv_' in p) 
        ] 
    eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested', coords='minimal').mean(
            'time', keep_attrs=True)
    fig, ax = plot_monthly_eml_occurence(eml_ds, fig, ax, month)
    if month == '10':
        ax.set_xticks(np.arange(-180, 240, 60))
        ax.set_xticklabels(['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'])
    axs.append(ax)
eml_ds.close()
# Annual cycle
n_eml_paths = np.sort(np.concatenate(
    [[p for p in glob.glob(
        base_path + f'{year}/monthly_means/geographical/'
        f'era5_3h_30N-S_eml_tropics_n_eml_{year}-*_geographical_mean.nc'
        )] for year in years], axis=0))
pfull_paths = np.sort(np.concatenate(
    [[p for p in glob.glob(
        base_path + f'{year}/monthly_means/geographical/'
        f'era5_3h_30N-S_eml_tropics_pfull_mean_{year}-*_geographical_mean.nc'
        )] for year in years], axis=0))
# iwv_paths = np.sort(np.concatenate(
#     [[p for p in glob.glob(
#         base_path + f'{year}/monthly_means/geographical/'
#         f'era5_3h_30N-S_eml_tropics_iwv_{year}-*_geographical_mean.nc'
#         )] for year in years], axis=0))
# rh_paths = np.sort(np.concatenate(
#     [[p for p in glob.glob(
#         base_path + f'{year}/monthly_means/geographical/'
#         f'era5_3h_30N-S_eml_tropics_rh_{year}-*_geographical_mean.nc'
#         )] for year in years], axis=0))
# rr_paths = np.sort(np.concatenate(
#     [[p for p in glob.glob(
#         base_path + f'{year}/monthly_means/geographical/'
#         f'era5_3h_30N-S_eml_tropics_rain_rate_{year}-*_geographical_mean.nc'
#         )] for year in years], axis=0))
time_coord = np.array([str(p)[-28:-21] for p in pfull_paths], dtype='datetime64[M]')
print("Loading global data of monthly means...")
eml_ds = xr.merge(
    [xr.open_mfdataset(paths, concat_dim='time', combine='nested') 
    for paths in [n_eml_paths, pfull_paths,]
    ]).assign_coords({'time': time_coord}).load()
print("Done loading data!")
# eml_ds_grouped = eml_ds.groupby('time.month')
# eml_ds_mean = eml_ds_grouped.mean('time')
# eml_ds_std = eml_ds_grouped.std('time')
lat_mesh, lon_mesh = np.meshgrid(eml_ds.lat, eml_ds.lon) 
is_ocean = xr.DataArray(
        globe.is_ocean(lat_mesh, lon_mesh), 
        dims=['lon', 'lat'])
eml_ds = eml_ds.where(is_ocean)

eml_ds_atlantic = eml_ds.sel(
    {'lon': slice(-60, 0)}
)
eml_ds_west_pacific = eml_ds.sel(
    {'lon': slice(120, 180)}
)
eml_ds_east_pacific = eml_ds.sel(
    {'lon': slice(-150, -90)}
)

n_eml_atlantic = eml_ds_atlantic.n_eml_0p3.where(
    (eml_ds_atlantic.pfull_mean > 50000) & (eml_ds_atlantic.pfull_mean < 70000)
    ).sum(['lat', 'lon', 'fulllevel'])
n_eml_west_pacific = eml_ds_west_pacific.n_eml_0p3.where(
    (eml_ds_west_pacific.pfull_mean > 50000) & (eml_ds_west_pacific.pfull_mean < 70000)
    ).sum(['lat', 'lon', 'fulllevel'])
n_eml_east_pacific = eml_ds_east_pacific.n_eml_0p3.where(
    (eml_ds_east_pacific.pfull_mean > 50000) & (eml_ds_east_pacific.pfull_mean < 70000)
    ).sum(['lat', 'lon', 'fulllevel'])

n_eml_mean_atlantic = n_eml_atlantic.groupby('time.month').mean('time')
n_eml_mean_east_pacific = n_eml_east_pacific.groupby('time.month').mean('time')
n_eml_mean_west_pacific = n_eml_west_pacific.groupby('time.month').mean('time')

n_eml_std_atlantic = n_eml_atlantic.groupby('time.month').std('time')
n_eml_std_east_pacific = n_eml_east_pacific.groupby('time.month').std('time')
n_eml_std_west_pacific = n_eml_west_pacific.groupby('time.month').std('time')
# n_eml_std_atlantic = eml_ds_std_atlantic.n_eml_0p3.where(
#     (eml_ds_std_atlantic.pfull_mean > 50000) & (eml_ds_std_atlantic.pfull_mean < 70000)
#     ).sum(['lat', 'lon', 'fulllevel'])
# n_eml_std_atlantic_std = eml_ds_std_atlantic.n_eml_0p3.where(
#     (eml_ds_std_atlantic.pfull_mean > 50000) & (eml_ds_std_atlantic.pfull_mean < 70000)
#     ).sum(['lat', 'lon', 'fulllevel'])
# n_eml_std_west_pacific = eml_ds_std_west_pacific.n_eml_0p3.where(
#     (eml_ds_std_west_pacific.pfull_mean > 50000) & (eml_ds_std_west_pacific.pfull_mean < 70000)
#     ).sum(['lat', 'lon', 'fulllevel'])
# n_eml_std_east_pacific = eml_ds_std_east_pacific.n_eml_0p3.where(
#     (eml_ds_std_east_pacific.pfull_mean > 50000) & (eml_ds_std_east_pacific.pfull_mean < 70000)
#     ).sum(['lat', 'lon', 'fulllevel'])

months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax = fig.add_subplot(gs[:, 1])
ax.plot(
    n_eml_mean_atlantic, months, label='atlantic', 
    color=sns.color_palette('colorblind')[0])
ax.fill_betweenx(
    months, 
    n_eml_mean_atlantic-n_eml_std_atlantic, 
    n_eml_mean_atlantic+n_eml_std_atlantic, 
    color=sns.color_palette('colorblind')[0], alpha=0.2, lw=0.2)
ax.plot(
    n_eml_mean_west_pacific, months, label='west pacific', 
    color=sns.color_palette('colorblind')[1], alpha=0.2)
ax.fill_betweenx(
    months, 
    n_eml_mean_west_pacific-n_eml_std_west_pacific, 
    n_eml_mean_west_pacific+n_eml_std_west_pacific, 
    color=sns.color_palette('colorblind')[1], alpha=0.2, lw=0.2)
ax.plot(
    n_eml_mean_east_pacific, months, label='east pacific', 
    color=sns.color_palette('colorblind')[2])
ax.fill_betweenx(
    months, 
    n_eml_mean_east_pacific-n_eml_std_east_pacific, 
    n_eml_mean_east_pacific+n_eml_std_east_pacific, 
    color=sns.color_palette('colorblind')[2], alpha=0.2, lw=0.2)

ax.hlines([0, 3, 6, 9], [1e6]*4, [4e6]*4, ls='--', color='black', lw=0.5)
ax.invert_yaxis()
ax.set(xlabel='EML count / -', xlim=[1e6, 4e6])
[ax.get_yticklabels()[i].set_weight('bold') for i in [0, 3, 6, 9]]
axs.append(ax)
label_axes(
    axs, labels=['a)', 'b)', 'c)', 'd)', 'e)'], fontsize=11, 
    loc=(0.02, 0.83))
plt.savefig(
    f'plots/revision/eml_occurence_annual_cycle_{years[0]}-{years[-1]}.pdf', 
    dpi=300, bbox_inches='tight')
