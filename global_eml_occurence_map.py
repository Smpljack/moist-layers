from turtle import filling
import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import eval_eml_chars as eec
import cartopy.crs as ccrs
from typhon.plots import worldmap

def main():
    month = '07'
    # Load gridded EML data
    eml_ds_paths = np.sort(
        glob.glob(
                '/home/u/u300676/user_data/mprange/eml_data/gridded/'
                'monthly_means/geographical/'
                f'gridded_monsoon_0p25deg_eml_tropics_*_2021-{month}_geographical_mean.nc'
                ))
    eml_ds_paths = [p for p in eml_ds_paths 
    if ('_rh_' not in p) 
    and ('_t_' not in p) 
    and ('_q_ref_' not in p)
    and ('_va_' not in p)
    and ('_zg_' not in p)
    ] 
                           

    eml_ds = xr.open_mfdataset(
        eml_ds_paths)
    # eml_ds = eec.filter_gridded_eml_ds(
    #     eml_ds,
    #     min_strength=0.3,
    #     min_pmean=10000,
    #     max_pmean=90000,
    #     min_pwidth=5000,
    #     max_pwidth=40000)
    
    # eml_ds_lower = eec.filter_gridded_eml_ds(
    #     eml_ds.copy(),
    #     min_strength=0.3,
    #     min_pmean=70000,
    #     max_pmean=90000,
    #     min_pwidth=5000,
    #     max_pwidth=40000)

    # eml_ds_middle = eec.filter_gridded_eml_ds(
    #     eml_ds.copy(),
    #     min_strength=0.3,
    #     min_pmean=50000,
    #     max_pmean=70000,
    #     min_pwidth=5000,
    #     max_pwidth=40000)

    # eml_ds_upper = eec.filter_gridded_eml_ds(
    #     eml_ds.copy(),
    #     min_strength=0.3,
    #     min_pmean=25000,
    #     max_pmean=45000,
    #     min_pwidth=5000,
    #     max_pwidth=40000)

    n_eml_lower = eml_ds.n_eml_0p3.where(
        (eml_ds.pfull_mean > 70000) & (eml_ds.pfull_mean < 90000)).sum('fulllevel')
    n_eml_middle = eml_ds.n_eml_0p3.where(
        (eml_ds.pfull_mean > 50000) & (eml_ds.pfull_mean < 70000)).sum('fulllevel')
    n_eml_upper = eml_ds.n_eml_0p3.where(
        (eml_ds.pfull_mean > 25000) & (eml_ds.pfull_mean < 45000)).sum('fulllevel')
    # n_eml_lower = (~np.isnan(eml_ds_lower.eml_strength)).sum(dim=['time', 'fulllevel']).values
    # n_eml_middle = (~np.isnan(eml_ds_middle.eml_strength)).sum(dim=['time', 'fulllevel']).values
    # n_eml_upper = (~np.isnan(eml_ds_upper.eml_strength)).sum(dim=['time', 'fulllevel']).values
    fig, axs = plt.subplots(
        nrows=3, subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(10, 7.5))
    s = worldmap(
            eml_ds.lat, eml_ds.lon, n_eml_upper/eml_ds.n_time, cmap='density',
            draw_coastlines=True, ax=axs[0], fig=fig, vmin=0, vmax=1
            )
    axs[0].set(title=f'2021-{month}, 250 - 450 hPa')
    s = worldmap(
            eml_ds.lat, eml_ds.lon, n_eml_middle/eml_ds.n_time, cmap='density',
            draw_coastlines=True, ax=axs[1], fig=fig, vmin=0, vmax=1,
            )
    axs[1].set(title=f'2021-{month}, 500 - 700 hPa')
    s = worldmap(
            eml_ds.lat, eml_ds.lon, n_eml_lower/eml_ds.n_time, cmap='density',
            draw_coastlines=True, ax=axs[2], fig=fig, vmin=0, vmax=1
            )
    axs[2].set(title=f'2021-{month}, 700 - 900 hPa')
    rr_levels = np.nanpercentile(
        eml_ds.rain_gsp_rate_mean*3600, [75, 95, 100])
    for i in range(len(axs)):
        axs[i].contourf(
            eml_ds.lon, eml_ds.lat, eml_ds.rain_gsp_rate_mean * 3600, 
            levels=rr_levels, alpha=0.4,
            vmin=rr_levels[0]-3,
            cmap='Reds', )
    cbar = plt.colorbar(
            s, orientation='horizontal', pad=0.05, aspect=50, extendrect=True, ax=axs[2])
    cbar.ax.set_xlabel('EML occurence rate / -')
    [ax.hlines([0, -15, 15], -180, 180, color='black', ls='--', lw=0.5) for ax in axs]
    [ax.vlines(
        [-60, 0, -150, -90, 120, 179.75], -30, 30, 
        color=['blue', 'blue', 'green', 'green', 'red', 'red'], 
        ls='-', lw=0.5) for ax in axs]
    [ax.set_yticks([-15, 0, 15]) for ax in axs]
    axs[0].set_xticks([-30, -120, 150])
    [ax.set_yticklabels([u'15\N{DEGREE SIGN}S', u'Eq', u'15\N{DEGREE SIGN}N']) for ax in axs]
    axs[0].set_xticklabels(
        ['Atlantic', 'East Pacifc', 'West Pacific'])
    [xtick.set_color(color) for xtick, color in
     zip(axs[0].get_xticklabels(), ['blue', 'green', 'red'])] 
    plt.savefig(
        f'plots/monsoon_gridded/monsoon_global_eml_occurence_lower_mid_upper_2021-{month}.png', dpi=300)

if __name__ == '__main__':
    main()