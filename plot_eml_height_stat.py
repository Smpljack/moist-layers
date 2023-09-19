import xarray as xr
import numpy as np
import intake
import argparse
import matplotlib.pyplot as plt

import eval_eml_chars as eec
import moist_layers as ml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_start", type=str,
                    help="timestamp",
                    default="2021-07-28T00:00:00")
    parser.add_argument("--time_end", type=str,
                    help="timestamp",
                    default="2021-08-11T00:00:00")
    args = parser.parse_args()
     # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]
    
    ds3d = cat["luk1043"].atm3d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    ds2d = cat["luk1043"].atm2d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_eurec4a(ds3d, grid)
    ds2d = ml.mask_eurec4a(ds2d, grid)
    grid = ml.mask_eurec4a(grid, grid)
    times = ds3d.time.values
    eml_ds = None
    for time in times:
        if eml_ds is None:
            eml_ds = eec.load_eml_data(time, 'extended_rh_def')  
        else:
            eml_ds = xr.concat([eml_ds, eec.load_eml_data(time, 'extended_rh_def')], dim='eml_count')
    eml_ds = eec.filter_eml_data(
        eml_ds, 
        min_strength=0.3, 
        min_pmean=10000, 
        max_pmean=90000, 
        min_pwidth=5000, 
        max_pwidth=40000,
        )
    fig, axs = plt.subplots()
    axs.hist(eml_ds.pmean/100, 30, label='Northern Atlantic', alpha=0.5)
    for time in times:
        if eml_ds is None:
            eml_ds = eec.load_eml_data(time, 'warmpool_rh_def')  
        else:
            eml_ds = xr.concat([eml_ds, eec.load_eml_data(time, 'warmpool_rh_def')], dim='eml_count')
    eml_ds = eec.filter_eml_data(
        eml_ds,
        min_strength=0.3, 
        min_pmean=10000, 
        max_pmean=90000, 
        min_pwidth=5000, 
        max_pwidth=40000,
        )
    axs.hist(eml_ds.pmean/100, 30, label='West Northern Pacific', alpha=0.5)
    axs.set(xlabel='EML pressure mean', ylabel='Count', 
            title='2021-07-28 to 2021-08-11')
    fig.legend()
    plt.savefig('test_plots/pmean_hist_atlantic_pacific.png', dpi=300)


if __name__ == '__main__':
    main()