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

def plot_monthly_eml_occurence(eml_ds, fig, ax, month):
    
    c1 = sns.color_palette('colorblind')[0]
    c2 = sns.color_palette('colorblind')[1]
    c3 = sns.color_palette('colorblind')[2]
    n_eml_middle = eml_ds.n_eml_0p3.where(
        (eml_ds.pfull_mean > 50000) & (eml_ds.pfull_mean < 70000)).sum('fulllevel')
    s = worldmap(
            eml_ds.lat, eml_ds.lon, n_eml_middle/eml_ds.n_time, cmap='density',
            draw_coastlines=True, ax=ax, fig=fig, vmin=0, vmax=1, 
            )
    ax.hlines([0, -15, 15], -180, 180, color='black', ls='--', lw=0.5)
    ax.vlines(
        [-60, 0, -150, -90, 120, 179.75], -30, 30, 
        color=[c1, c1, c2, c2, c3, c3], 
        ls='-', lw=1)
    ax.set_yticks([-15, 0, 15])
    ax.set_yticklabels([u'15\N{DEGREE SIGN}S', u'Eq', u'15\N{DEGREE SIGN}N'])
    if month == '01':
        ax.set_xticks([-30, -120, 150])
        ax.set_xticklabels(
            ['Atlantic', 'East Pacific', 'West Pacific'])
        ax.xaxis.set_ticks_position('top')
        [xtick.set_color(color) for xtick, color in 
            zip(ax.get_xticklabels(), [c1, c2, c3])] 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.13, axes_class=maxes.Axes)
    plt.axis('off')
    if month == '10':
        cbar = plt.colorbar(
        s, orientation='horizontal', pad=10, aspect=50, extendrect=True, shrink=0.86, 
        ax=cax)
        cbar.ax.set_xlabel('EML occurence rate / -')
    rr_levels = np.nanpercentile(
        eml_ds.rain_rate_mean*3600, [75, 95, 100])
    ax.contourf(
            eml_ds.lon, eml_ds.lat, eml_ds.rain_rate_mean * 3600, 
            levels=rr_levels, alpha=0.4,
            vmin=rr_levels[0]-3,
            cmap='Reds',)
    return fig, ax
    

def main():
    fig = plt.figure(
            figsize=(10, 5))
    gs = gridspec.GridSpec(
        nrows=4, ncols=2, width_ratios=[3, 1], hspace=0.001, wspace=0.1)
    axs = []
    for imonth, month in enumerate(['01', '04', '07', '10']):
        ax = fig.add_subplot(
            gs[imonth, 0], projection=ccrs.PlateCarree(central_longitude=0))
        # Load gridded EML data
        eml_ds_paths = np.sort(
            glob.glob(
                    '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                    'monthly_means/geographical/'
                    f'era5_3h_30N-S_eml_tropics_*_2021-{month}_geographical_mean.nc'
                    ))
        eml_ds_paths = [
            p for p in eml_ds_paths 
            if ('_pfull_' in p) 
            or ('_n_eml_' in p)
            or ('_rain_rate_' in p)
            ] 
        eml_ds = xr.open_mfdataset(
            eml_ds_paths)

        fig, ax = plot_monthly_eml_occurence(eml_ds, fig, ax, month)
        axs.append(ax)
    # Annual cycle
    n_eml_paths = np.sort(glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/geographical/'
                'era5_3h_30N-S_eml_tropics_n_eml_2021-*_geographical_mean.nc'
                ))
    pfull_paths = np.sort(glob.glob(
                    '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                    'monthly_means/geographical/'
                    f'era5_3h_30N-S_eml_tropics_pfull_2021-*_geographical_mean.nc'
                    ))
    # rr_paths = np.sort(glob.glob(
    #                 '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
    #                 'monthly_means/geographical/'
    #                 f'era5_3h_30N-S_eml_tropics_rain_rate_2021-*_geographical_mean.nc'
    #                 ))
    eml_ds = xr.merge(
        [
            xr.open_mfdataset(n_eml_paths, concat_dim='time', combine='nested'),
            xr.open_mfdataset(pfull_paths, concat_dim='time', combine='nested'),
            # xr.open_mfdataset(rr_paths, concat_dim='time', combine='nested')
        ])
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
    # sigma_iwv_atlantic = eml_ds_atlantic.iwv_mean.std(['lat', 'lon'])
    # sigma_iwv_west_pacific = eml_ds_west_pacific.iwv_mean.std(['lat', 'lon'])
    # sigma_iwv_east_pacific = eml_ds_east_pacific.iwv_mean.std(['lat', 'lon'])
    # iorg_atlantic = []
    # for imonth in range(12):
    #     rr_thresh = np.nanpercentile(
    #         eml_ds_atlantic.rain_rate_mean.isel(time=imonth), 95)
    #     conv_array = get_convective_array(
    #         eml_ds_atlantic.rain_rate_mean.isel(time=imonth), rr_thresh)
    #     iorg_atlantic.append(iorg(conv_array))
    # months = np.arange('2021-01', '2021-12', dtype='datetime64[1M]')
    months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # month_labels = [
    #     r'\textbf{Jan}', 'Feb', 'Mar', r'\textbf{Apr}', 'May', 'Jun', 
    #     r'\textbf{Jul}', 'Aug', 'Sep', r'\textbf{Oct}', 'Nov']
    ax = fig.add_subplot(gs[:, 1])
    ax.plot(
        n_eml_atlantic, months, label='atlantic', 
        color=sns.color_palette('colorblind')[0])
    ax.plot(
        n_eml_west_pacific, months, label='west pacific', 
        color=sns.color_palette('colorblind')[1])
    ax.plot(
        n_eml_east_pacific, months, label='east pacific', 
        color=sns.color_palette('colorblind')[2])
    # ax2 = ax.twiny()
    # ax2.plot(
    #     iorg_atlantic, months, 
    #     color=sns.color_palette('colorblind')[0], ls='--')
    # ax2.plot(
    #     sigma_iwv_west_pacific, months, 
    #     color=sns.color_palette('colorblind')[1], ls='--')
    # ax2.plot(
    #     sigma_iwv_east_pacific, months, 
    #     color=sns.color_palette('colorblind')[2], ls='--')
    # ax2.set(xlabel=r'$\sigma$(IWV)$^{2}$ / kg$^{2}$ m$^{-4}$')
    # ax2.set(xlabel=r'I$_{org}$')
    ax.hlines([0, 3, 6, 9], [1e6]*4, [4e6]*4, ls='--', color='black', lw=0.5)
    ax.invert_yaxis()
    ax.set(xlabel='EML count / -', xlim=[1e6, 4e6])
    [ax.get_yticklabels()[i].set_weight('bold') for i in [0, 3, 6, 9]]
    # ax.legend()
    axs.append(ax)
    label_axes(
        axs[:-1], labels=['a)', 'b)', 'c)', 'd)'], fontsize=11, 
        loc=(0.02, 0.83))
    label_axes(
        [axs[-1]], labels=['e)'], fontsize=11, 
        loc=(0.02, 0.97)) 
    plt.savefig('plots/paper/eml_occurence_annual_cycle.png', dpi=300)

if __name__ == '__main__':
    main()