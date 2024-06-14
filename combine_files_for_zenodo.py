import xarray as xr

inpath = '/work/um0878/user_data/mprange/era5_tropics/eml_data/'
outpath = inpath+'zenodo/'

months = ['01', '07']

geo_vars = ['eml_pmax', 'eml_pmin', 'eml_pmean', 'n_eml', 'pfull_mean']
ms_vars = ['n_eml', 'pfull']

# for year in range(2009, 2024):
#     for month in months:
#         geo_cmb = xr.merge(
#                 [xr.open_dataset(
#                     f'{inpath}{year}/monthly_means/geographical/'
#                     f'era5_3h_30N-S_eml_tropics_{var}_{year}-{month}_geographical_mean.nc')
#                 for var in geo_vars]).isel(fulllevel=slice(5, -1))
#         ms_cmb = xr.merge(
#                 [xr.open_dataset(
#                     f'{inpath}{year}/monthly_means/moisture_space/'
#                     f'era5_3h_30N-S_eml_tropics_atlantic_{var}_{year}-{month}_moisture_space_mean.nc')
#              for var in ms_vars]).isel(fulllevel=slice(5, -1))
#         hm_cmb = xr.concat(
#                 [xr.open_dataset(
#                     f'{inpath}{year}/'
#                     f'era5_3h_atlantic_{lat}N_{year}-{month}.nc') 
#                 for lat in [0, 15]], dim='lat').isel(fulllevel=slice(5, -1))

#         drop_geo_vars = [var for var  in geo_cmb.variables if ('std' in var or 'mean_of_squares' in var)]
#         drop_ms_vars = [var for var in ms_cmb.variables if ('std' in var or 'mean_of_squares' in var)]
#         drop_hm_vars = [var for var in hm_cmb.variables if ('eml' not in var and var not in ['pfull'])]
#         for var in drop_geo_vars:
#             geo_cmb = geo_cmb.drop(var)
#         for var in drop_ms_vars:
#             ms_cmb = ms_cmb.drop(var)
#         for var in drop_hm_vars:
#             hm_cmb = hm_cmb.drop(var)
#         geo_cmb.to_netcdf(f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_{year}-{month}_geographical_mean.nc')
#         ms_cmb.to_netcdf(f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_{year}-{month}_moisture_space_mean.nc')
#         hm_cmb.to_netcdf(f'{inpath}zenodo/era5_3h_atlantic_xsec_hovmoller_{year}-{month}.nc')

# Multi-year means
for month in months:
    geo_cmb = xr.open_mfdataset(
                f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_*-{month}_geographical_mean.nc',
                combine='nested', concat_dim='time')
    geo_cmb_mean = geo_cmb.mean('time')
    geo_cmb_std = geo_cmb.std('time')
    ms_cmb = xr.open_mfdataset(
                f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_*-{month}_moisture_space_mean.nc',
                combine='nested', concat_dim='time')
    ms_cmb_mean = ms_cmb.mean('time')
    ms_cmb_std = ms_cmb.std('time')
    geo_cmb_mean.to_netcdf(
        f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_mean_2009-2023-{month}_geographical.nc')
    ms_cmb_mean.to_netcdf(
        f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_mean_2009-2023-{month}_moisture_space.nc')
    geo_cmb_std.to_netcdf(
        f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_std_2009-2023-{month}_geographical.nc')
    ms_cmb_std.to_netcdf(
        f'{inpath}zenodo/era5_3h_30N-S_eml_tropics_eml_vars_std_2009-2023-{month}_moisture_space.nc')
