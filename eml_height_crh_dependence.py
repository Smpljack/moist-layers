import xarray as xr
import numpy as np
import intake
import argparse
import matplotlib.pyplot as plt
import glob
import global_land_mask as globe
import matplotlib.gridspec as gridspec

from typhon.plots import heatmap
import eval_eml_chars as eec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str,
                    help="timestamp",
                    default="07")
    args = parser.parse_args()
    eml_ds_paths_global = glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/moisture_space/'
                f'era5_3h_30N-S_eml_tropics_*_2021-{args.month}_moisture_space_mean.nc'
                )
    eml_ds_paths_global = [p for p in eml_ds_paths_global 
                           if ('atlantic' not in p) 
                           and ('west_pacific' not in p)
                           and ('east_pacific' not in p)]
    eml_ds_paths_atlantic = glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/moisture_space/'
                f'era5_3h_30N-S_eml_tropics_atlantic_*_2021-{args.month}_moisture_space_mean.nc'
                )
    eml_ds_paths_west_pacific = glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/moisture_space/'
                f'era5_3h_30N-S_eml_tropics_west_pacific_*_2021-{args.month}_moisture_space_mean.nc'
                )
    eml_ds_paths_east_pacific = glob.glob(
                '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
                'monthly_means/moisture_space/'
                f'era5_3h_30N-S_eml_tropics_east_pacific_*_2021-{args.month}_moisture_space_mean.nc'
                )
    eml_ds_global = xr.open_mfdataset(eml_ds_paths_global)
    eml_ds_atlantic = xr.open_mfdataset(eml_ds_paths_atlantic)
    eml_ds_west_pacific = xr.open_mfdataset(eml_ds_paths_west_pacific)
    eml_ds_east_pacific = xr.open_mfdataset(eml_ds_paths_east_pacific)


    eml_iwv_global = eml_ds_global.iwv_bins
    pfull_global = eml_ds_global.pfull_mean
    n_eml_global = eml_ds_global.n_eml_0p3.where(pfull_global < 80000)
    rh_mean_global = eml_ds_global.rh_mean.mean('iwv_bins')
    rh_std_global = eml_ds_global.rh_std.mean('iwv_bins')
    pfull_mean_global = pfull_global.mean('iwv_bins')

    eml_iwv_atlantic = eml_ds_atlantic.iwv_bins
    pfull_atlantic = eml_ds_atlantic.pfull_mean
    n_eml_atlantic = eml_ds_atlantic.n_eml_0p3.where(pfull_atlantic < 80000)
    rh_mean_atlantic = eml_ds_atlantic.rh_mean#.mean('iwv_bins')
    rh_std_atlantic = eml_ds_atlantic.rh_std#.mean('iwv_bins')
    pfull_mean_atlantic = pfull_atlantic#.mean('iwv_bins')

    eml_iwv_west_pacific = eml_ds_west_pacific.iwv_bins
    pfull_west_pacific = eml_ds_west_pacific.pfull_mean
    n_eml_west_pacific = eml_ds_west_pacific.n_eml_0p3.where(pfull_west_pacific < 80000)
    rh_mean_west_pacific = eml_ds_west_pacific.rh_mean.mean('iwv_bins')
    rh_std_west_pacific = eml_ds_west_pacific.rh_std.mean('iwv_bins')
    pfull_mean_west_pacific = pfull_west_pacific.mean('iwv_bins')

    eml_iwv_east_pacific = eml_ds_east_pacific.iwv_bins
    pfull_east_pacific = eml_ds_east_pacific.pfull_mean
    n_eml_east_pacific = eml_ds_east_pacific.n_eml_0p3.where(pfull_east_pacific < 80000)
    rh_mean_east_pacific = eml_ds_east_pacific.rh_mean.mean('iwv_bins')
    rh_std_east_pacific = eml_ds_east_pacific.rh_std.mean('iwv_bins')
    pfull_mean_east_pacific = pfull_east_pacific.mean('iwv_bins')
    percentiles = np.arange(1, 100, 2)
    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6], hspace=0.1)
    # s = axs[0].pcolormesh(
    #     percentiles, pfull_global/100, n_eml_global, cmap='density', )
    # axs[0].set(xlabel='percentile of IWV', ylabel='pressure / hPa',
    #         title='Global', ylim=[1020, 100], xlim=[0, 100])
    # cb = plt.colorbar(
    #     s, orientation='horizontal', extend='max', ax=axs[0], pad=0.08)
    # cb.ax.set(xlabel='count')

    ax1 = fig.add_subplot(gs[1, 0])
    s = ax1.pcolormesh(
        percentiles, 
        pfull_atlantic.where(pfull_atlantic < 80000).dropna('fulllevel')/100, 
        (n_eml_atlantic / 
         eml_ds_atlantic.rh_count.where(pfull_atlantic < 80000)
         ).dropna('fulllevel'), 
         cmap='density', 
         vmin=0,
         vmax=0.1,)
    ax1.set(xlabel='percentile of IWV', ylabel='pressure / hPa',
            ylim=[1020, 100], xlim=[0, 100])
    cb = plt.colorbar(
        s, orientation='horizontal', extend='max', ax=ax1, pad=0.15)
    cb.ax.set(xlabel='EML fraction')
    percentiles_2d = np.tile(
    np.arange(1, 100, 2), 
    len(eml_ds_atlantic.fulllevel)).reshape((len(eml_ds_atlantic.fulllevel), 50)).T
    # c1 = ax1.contour(
    #     percentiles_2d.T, 
    #     pfull_atlantic/100, 
    #     rh_std_atlantic.where(
    #     (pfull_atlantic < 80000) & (pfull_atlantic > 25000))*100,
    #     levels=np.arange(0, 32, 2),
    #     vmin=0, vmax=30,
    #     cmap='Greys', alpha=1,
    #     extend='max')
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    ax2.plot(
        rh_mean_atlantic.isel(
            iwv_bins=slice(25, 45)).mean('iwv_bins')*100, 
        pfull_mean_atlantic.isel(
            iwv_bins=slice(25, 45)).mean('iwv_bins')/100, 
        label=r'$\overline{\mathrm{RH}}$', color='black', )
    ax2.plot(
        rh_std_atlantic.isel(
            iwv_bins=slice(25, 45)).mean('iwv_bins')*100, 
            pfull_mean_atlantic.isel(
            iwv_bins=slice(25, 45)).mean('iwv_bins')/100, 
         color='black', ls='--',
        label=r'$\sigma$(RH)')
    ax2.set(
        xlabel='RH / %', ylim=[1020, 100], xlim=[0, 100], )
    ax2.legend()
    pos0 = ax1.get_position()
    pos1 = ax2.get_position()
    ax2.set_position([pos1.x0, pos0.y0, pos1.width, pos0.height])
    # ax2.set_yticklabels([])
    ax3 = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax3.plot(
        percentiles, 
        n_eml_atlantic.sum('fulllevel') / 
        eml_ds_atlantic.rh_count[50, :],
        color='black')
    ax3.plot([0, 100], [1, 1], '--k', lw=0.5)
    ax3.set(ylim=[0, 1.2], ylabel='EML fraction',)
    # ax3.set_xticklabels([])

    # s = axs[2].pcolormesh(
    #     percentiles, pfull_west_pacific/100, n_eml_west_pacific, 
    #     cmap='density', )
    # axs[2].set(xlabel='percentile of IWV', 
    #         title='West Pacific', ylim=[1020, 100], xlim=[0, 100])
    # cb = plt.colorbar(
    #     s, orientation='horizontal', extend='max', ax=axs[2], pad=0.08)
    # cb.ax.set(xlabel='count')

    
    # s = axs[3].pcolormesh(
    #     percentiles, pfull_east_pacific/100, n_eml_east_pacific, 
    #     cmap='density', )
    # axs[3].set(xlabel='percentile of IWV', 
    #         title='East Pacific', ylim=[1020, 100], xlim=[0, 100])
    # cb = plt.colorbar(
    #     s, orientation='horizontal', extend='max', ax=axs[3], pad=0.08)
    # cb.ax.set(xlabel='count')
    # axs[4].plot(
    #     rh_mean_global*100, pfull_mean_global/100,
    #     label='Global', color='black')
    # axs[4].plot(
    #     rh_mean_atlantic*100, pfull_mean_atlantic/100, 
    #     label='Atlantic', color='blue')
    # axs[4].plot(
    #     rh_mean_west_pacific*100, pfull_mean_west_pacific/100, 
    #     label='West Pacific', color='red')
    # axs[4].plot(
    #     rh_mean_east_pacific*100, pfull_mean_east_pacific/100,
    #     label='East Pacific', color='green')
    # axs[4].plot(
    #     rh_std_global*100, pfull_mean_global/100,
    #     color='black', ls='--')
    # axs[4].plot(
    #     rh_std_atlantic*100, pfull_mean_atlantic/100, 
    #     color='blue', ls='--')
    # axs[4].plot(
    #     rh_std_west_pacific*100, pfull_mean_west_pacific/100, 
    #     color='red', ls='--')
    # axs[4].plot(
    #     rh_std_east_pacific*100, pfull_mean_east_pacific/100,
    #     color='green', ls='--')
    # axs[4].set(xlabel='RH / %', 
    #         title='mean RH profiles', ylim=[1020, 100], xlim=[0, 100])
    # pos3 = axs[3].get_position()
    # pos4 = axs[4].get_position()
    # axs[4].set_position([pos4.x0, pos3.y0, pos4.width, pos3.height])
    # axs[4].legend()
    plt.savefig(
        'plots/era5/'
        f'era5_eml_iwv_pmean_heatmap_profiles_atlantic_monthly_stat_{args.month}_2021_poster.png', dpi=300)


if __name__ == '__main__':
    main()