import xarray as xr
import numpy as np
import glob 
import global_land_mask as globe
import argparse
from pathlib import Path

import eval_eml_chars as eec

def squared_data_array(da):
    return np.square(da)

def n_emls(da):
    return (~da.isnull()).sum(['stacked_time_lat_lon'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default="2009-07")
    parser.add_argument("--variable", type=str,
                    help="variable name",
                    default="all")
    parser.add_argument("--region", type=str,
                        help="variable name",
                        default="global")
    parser.add_argument("--stat_type", type=str,
                        help="'geographical', 'moisture_space', 'hovmoller_xsec' or 'all'",
                        default="hovmoller_xsec")                  
    args = parser.parse_args()
    var = args.variable
    base_path = f'/home/u/u300676/user_data/mprange/era5_tropics/eml_data/{args.time[:4]}/'
    file_name_head = 'era5_3h_30N-S_eml_tropics_'
    paths = sorted(glob.glob(
        base_path + 
        file_name_head + f'{args.time}*[0-9].nc'
        ))
    print(f"Opening dataset for {args.time}", flush=True)
    eml_ds = xr.open_mfdataset(
        paths, concat_dim='time', combine='nested')
    eml_ds = eec.filter_gridded_eml_ds(
        eml_ds.copy(),
        min_strength=0.3,
        min_pmean=10000,
        max_pmean=90000,
        min_pwidth=5000,
        max_pwidth=40000)
    n = len(eml_ds.time)
    if args.stat_type in ['hovmoller_xsec', 'all'] and var == 'all' and args.region == 'global':
        eml_ds_15 = eml_ds.sel(
            {
                'lat': 15,
                'lon': slice(-60, 20),
            })
        eml_ds_eq = eml_ds.sel(
            {
                'lat': 0,
                'lon': slice(-60, 20),
            })
        eml_ds_15.to_netcdf(base_path + f'era5_3h_atlantic_15N_{args.time}.nc', encoding={"time": {"dtype": int}})
        eml_ds_eq.to_netcdf(base_path + f'era5_3h_atlantic_0N_{args.time}.nc', encoding={"time": {"dtype": int}})
        
    if args.stat_type in ['geographical', 'all'] and args.region == 'global' and var != 'all':
        if var != 'n_eml':
            print("Calculating geographical mean.", flush=True)
            mean = eml_ds[var].mean('time').rename(f'{var}_mean').load()
            print("Calculating geographical std.", flush=True)
            std = eml_ds[var].std('time').rename(f'{var}_std').load()  
            print("Calculating geographical mean of squares.", flush=True)
            mean_of_squares = (eml_ds[var]**2).mean(
                'time').rename(f'{var}_mean_of_squares').load()
            print("Merging geographical mean variables.", flush=True)
            mean_lat_lon_ds = xr.merge(
                [mean, std, mean_of_squares]).assign_attrs(
                    {'n_time': n}
                )

        if var == 'n_eml':
            print("Calculating geographical moist layer count.", flush=True)
            n_eml = (~eml_ds.eml_strength.isnull()).sum('time').rename(
                'n_eml_0p3').load()
            mean_lat_lon_ds = xr.merge(
                [n_eml]).assign_attrs(
                    {'n_time': n}
                )
        print("Storing geographical mean.", flush=True)
        dir = base_path + f'monthly_means/geographical/'
        Path(dir).mkdir(parents=True, exist_ok=True)
        mean_lat_lon_ds.to_netcdf(
            dir + file_name_head + f'{var}_{args.time}_geographical_mean.nc')

    if args.stat_type in ['moisture_space', 'all'] and var != 'all': 
        # Moisture space statistics
        print("Preparing moisture space data.", flush=True)
        if args.region == 'atlantic':
            eml_ds = eml_ds.sel(
                {'lon': slice(-60, 0)}
            )
        elif args.region == 'west_pacific':
            eml_ds = eml_ds.sel(
                {'lon': slice(120, 180)}
            )
        elif args.region == 'east_pacific':
            eml_ds = eml_ds.sel(
                {'lon': slice(-150, -90)}
            )
        lat_mesh, lon_mesh = np.meshgrid(eml_ds.lat, eml_ds.lon) 
        is_ocean = xr.DataArray(
                globe.is_ocean(lat_mesh, lon_mesh), 
                dims=['lon', 'lat'])
        eml_ds = eml_ds.where(is_ocean)

        percentiles = np.arange(0, 102, 2)
        iwv_bins = np.nanpercentile(eml_ds.iwv, percentiles)
        iwv_bin_labels = (iwv_bins[0:-1] + iwv_bins[1:])/2
        print("Grouping data by IWV.", flush=True)
        if var == 'n_eml':
            load_var = 'eml_strength'
        else: 
            load_var = var
        eml_ds_grouped = eml_ds[load_var].groupby_bins(
            eml_ds.iwv, iwv_bins, labels=iwv_bin_labels) 
        if var != 'n_eml':
            eml_ds_squared_grouped = (eml_ds[load_var]**2).groupby_bins(
                eml_ds.iwv, iwv_bins, labels=iwv_bin_labels)
            print("Calculating moisture space mean.", flush=True)
            eml_ds_iwv_grouped_mean = eml_ds_grouped.mean().rename(
                f'{var}_mean').load()
            print("Calculating moisture space std.", flush=True)
            eml_ds_iwv_grouped_std = eml_ds_grouped.std().rename(
                f'{var}_std').load()
            print("Calculating moisture space count.", flush=True)
            eml_ds_iwv_grouped_count = eml_ds_grouped.count().rename(
                f'{var}_count').load()
            print("Calculating moisture space mean of squares.", flush=True)
            eml_ds_iwv_grouped_mean_of_squares = eml_ds_squared_grouped.mean().rename(
                f'{var}_mean_of_squares').load()
            moisture_space_mean_ds = xr.merge(
                    [eml_ds_iwv_grouped_mean, eml_ds_iwv_grouped_std, 
                    eml_ds_iwv_grouped_count, eml_ds_iwv_grouped_mean_of_squares]) 
        
        if var == 'n_eml':
            print("Calculating moisture space moist layer count.", flush=True)
            eml_da_iwv_grouped_eml_count = eml_ds_grouped.apply(n_emls).rename(
                'n_eml_0p3').load() 
            print("Merging moisture space mean variables.", flush=True)
            moisture_space_mean_ds = xr.merge(
                [eml_da_iwv_grouped_eml_count])
            
        print("Storing moisture space mean.", flush=True)
        dir = base_path + f'monthly_means/moisture_space/'
        Path(dir).mkdir(parents=True, exist_ok=True)
        moisture_space_mean_ds.to_netcdf(
            dir + file_name_head + f'{args.region}_{var}_{args.time}_moisture_space_mean.nc') 


if __name__ == '__main__':
    main()
