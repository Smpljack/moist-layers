import xarray as xr
import numpy as np
import intake
import argparse
import matplotlib.pyplot as plt
import glob
import global_land_mask as globe

import eval_eml_chars as eec
import moist_layers as ml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2021-07-28")
    args = parser.parse_args()
    eml_ds_paths = sorted(
        glob.glob('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                  'gridded_monsoon_0p25deg_eml_tropics_2021-*_crh.nc'))
    eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested')
    eml_ds = eec.filter_gridded_eml_ds(
            eml_ds, 
            min_strength=0.3, 
            min_pmean=10000, 
            max_pmean=90000, 
            min_pwidth=5000, 
            max_pwidth=40000)
    eml_ds = eml_ds.rename({'lon': 'longitude', 'lat': 'latitude'})
    lat_mesh, lon_mesh = np.meshgrid(eml_ds.latitude, eml_ds.longitude) 
    is_ocean = xr.DataArray(
        globe.is_ocean(lat_mesh, lon_mesh), 
        dims=['longitude', 'latitude'])
    eml_ds = eml_ds.where(is_ocean)
    eml_ds_atlantic = eml_ds.sel(
        {'longitude': slice(-60, 0)}
    )
    eml_ds_east_pacific = eml_ds.sel(
        {'longitude': slice(-180, -120)}
    )
    eml_ds_west_pacific = eml_ds.sel(
        {'longitude': slice(120, 180)}
    )
    # eml_ds_warmpool = xr.concat(
    #     [
    #         eml_ds.sel(
    #         {'longitude': slice(150, 180)}
    #     ),
    #         eml_ds.sel(
    #         {'longitude': slice(-180, -150)}
    #     ) 
    #     ],
    #     dim='longitude')
    eml_pmean_global = eml_ds.pfull.where(
        ~np.isnan(eml_ds.eml_strength)
    ).values.flatten()
    eml_pmean_global = eml_pmean_global[~np.isnan(eml_pmean_global)]
    eml_pmean_atlantic = eml_ds_atlantic.pfull.where(
        ~np.isnan(eml_ds_atlantic.eml_strength)
    ).values.flatten()
    eml_pmean_atlantic = eml_pmean_atlantic[~np.isnan(eml_pmean_atlantic)]
    eml_pmean_west_pacific = eml_ds_west_pacific.pfull.where(
        ~np.isnan(eml_ds_west_pacific.eml_strength)
    ).values.flatten()
    eml_pmean_west_pacific = eml_pmean_west_pacific[~np.isnan(eml_pmean_west_pacific)]
    eml_pmean_east_pacific = eml_ds_east_pacific.pfull.where(
        ~np.isnan(eml_ds_east_pacific.eml_strength)
    ).values.flatten()
    eml_pmean_east_pacific = eml_pmean_east_pacific[~np.isnan(eml_pmean_east_pacific)]
    eml_pfull_mean = eml_ds_east_pacific.pfull.mean(
        ['latitude', 'longitude', 'time'])
    p_bins = np.concatenate(
        [(eml_pfull_mean[48:-1] - np.diff(eml_pfull_mean[48:])).values,
         [eml_pfull_mean.values[-1]]]) / 100
    p_bins = p_bins[(p_bins < 900) & (p_bins > 200)]
    fig, axs = plt.subplots()
    _, _, h1 = axs.hist(
        eml_pmean_atlantic/100, bins=p_bins, label='Atlantic', alpha=0.5,
        histtype='step', linewidth=2)
    _, _, h2 = axs.hist(
        eml_pmean_west_pacific/100, bins=p_bins, label='West Pacific', alpha=0.5,
        histtype='step', linewidth=2)
    _, _, h3 = axs.hist(
        eml_pmean_east_pacific/100, bins=p_bins, label='East Pacific', alpha=0.5,
        histtype='step', linewidth=2)
    ax2 = axs.twinx()
    _, _, h4 = ax2.hist(
        eml_pmean_global/100, bins=p_bins, label='Global', alpha=0.5,
        histtype='step', linewidth=2, color='black')
    ax2.set(ylabel='Global count', ylim=[0, 70000*14])
    axs.set(xlabel='EML pressure mean / hPa', ylabel='Regional count',
            title='2021-07-28 to 2021-08-11', ylim=[0, 30000*14])
    handles = [h1[0], h2[0], h3[0] , h4[0]] 
    plt.legend(
        handles, [h.get_label() for h in handles],
        loc='upper right')
    plt.savefig(
        'plots/monsoon_gridded/'
        'monsoon_eml_pmean_hist_atlantic_WE_pacific_gridded_ocean.png', 
        dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()