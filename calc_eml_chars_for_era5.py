from multiprocessing.sharedctypes import Value
import numpy as np
import xarray as xr
import argparse
import glob
import multiprocessing

from typhon.physics import (specific_humidity2vmr, 
                            vmr2relative_humidity, 
                            e_eq_mixed_mk, pressure2height,
                            vmr2specific_humidity)

import moist_layers as ml
import eval_eml_chars as eec
from add_var_to_ds import add_iwv_to_ds

def prepare_era5_ds(lnsp, q, t, u=None, v=None, w=None, rr=None):
    ab_params = xr.open_dataset('~/era5_processing/ifs_p_params.nc')
    phalf = (np.exp(lnsp.sel(lev=1.)) * ab_params.b_mid + ab_params.a_mid)
    pfull = 0.5 * (
        phalf[:, :, :-1] + phalf.values[:, :,1:]
        ).transpose(
            'lev', 'latitude', 'longitude'
            ).assign_coords({'lev': np.arange(1, 138)})
    vmr_h2o = specific_humidity2vmr(q)
    rh = vmr2relative_humidity(
        vmr_h2o.values, pfull.values, t.values, e_eq=e_eq_mixed_mk) 
    z = np.nan * np.ones(pfull.shape)
    for lat in range(len(pfull.latitude)):
        print('calc z for\n'
            f'lat {pfull.latitude[lat].values}\n', flush=True)
        for lon in range(len(pfull.longitude)): 
            z[:, lat, lon] = pressure2height(
                pfull.values[::-1, lat, lon], 
                t.values[::-1, lat, lon])[::-1]
    merge_vars = [var for var in [q, t, u, v, w, rr] if var is not None]
    era5_ds = xr.merge(merge_vars).assign(
        variables={
                'pfull': pfull,
                'vmr_h2o': vmr_h2o,
                'rh': (('lev', 'latitude', 'longitude'), rh),
                'z': (('lev', 'latitude', 'longitude'), z)
            }
    ).rename(
        {
            'latitude': 'lat',
            'longitude': 'lon',
            'lev': 'fulllevel',
        }
    ).isel(fulllevel=slice(42, 138))
    rename = {'var228': 'rain_rate'}
    for var in list(era5_ds.variables):
        if var in ['var228']:
            era5_ds = era5_ds.rename({var: rename[var]})
    return era5_ds

def store_eml_ds_multi_proc(lnsp, q, t, u, v, w, rr, time):
    queue = multiprocessing.Queue()
    process_time_bounds = np.arange(
        lnsp.time[0].values.astype('datetime64[D]'), 
        lnsp.time[-1].values.astype('datetime64[D]')+np.timedelta64(1, 'D'))

    processes = []
    for ptime in process_time_bounds:
        print(f'Started process for {ptime}', flush=True)
        p = multiprocessing.Process(
            target=calc_eml_ds_for_day, 
            args=(lnsp, q, t, u, v, w, rr),
            kwargs={'queue': queue, 'day': ptime})
        processes.append(p)
        p.start()

    # print('Concat.', flush=True)
    # eml_ds_list = [queue.get() for p in processes]
    for p in processes:
        print('Joining process.', flush=True)
        p.join()
    # print('Concatenating data...', flush=True)
    # eml_ds = xr.concat(eml_ds_list, dim='time').sortby('time')
    # print("Storing data", flush=True)
    # for eml_ds in eml_ds_list:
    #     eml_ds.to_netcdf(
    #         '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
    #         f'era5_3h_30N-S_eml_tropics_{str(eml_ds.time.values[0].astype("datetime64[D]"))}.nc')

def calc_eml_ds_for_day(
    lnsp, q, t, u, v, w, rr, 
    queue=None, day=None):
    """"""
    if day is not None:
        lnsp = lnsp.sel(time=str(day))
        q = q.sel(time=str(day))
        t = t.sel(time=str(day))
        u = u.sel(time=str(day))
        v = v.sel(time=str(day))
        w = w.sel(time=str(day))
        rr = rr.sel(time=str(day))
    times = lnsp.time
    # era5_1m_ds = None
    
    for time in times:
        print(f'calc eml characteristics for \n{time.values}', flush=True)
        lnsp_t = lnsp.sel(time=time).lnsp.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})
        q_t = q.sel(time=time).q.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})
        t_t = t.sel(time=time).t.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})
        u_t = u.sel(time=time).u.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})
        v_t = v.sel(time=time).v.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})
        w_t = w.sel(time=time).w.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})
        rr_t = rr.sel(time=time).var228.load()#.isel({'longitude': slice(0, 10), 'latitude': slice(0, 10)})

        era5_1h_ds = prepare_era5_ds(lnsp_t, q_t, t_t, u=u_t, v=v_t, w=w_t, rr=rr_t)
        era5_1h_ds = era5_1h_ds.assign(
            {
                f'{eml_var}': (('fulllevel', 'lat', 'lon'), 
                                np.full(era5_1h_ds.q.shape, np.nan))
                for eml_var in 
                ['eml_strength', 'eml_pmean', 'eml_pwidth', 
                'eml_pmax', 'eml_pmin', #'eml_zmean', 'eml_zmax', 'eml_zmin', 
                'q_ref']
            }
        )
        for ilat, lat in enumerate(era5_1h_ds.lat.values):
            print('calc eml characteristics for\n'
                f'lat {lat}\n {time.values}\n', flush=True)
            for ilon, lon in enumerate(era5_1h_ds.pfull.lon.values):
                era5_1h_ds_col = era5_1h_ds.sel(
                    {'lat': lat, 'lon': lon}) 
                vmr_h2o_col = era5_1h_ds_col.vmr_h2o.values[::-1]
                t_col = era5_1h_ds_col.t.values[::-1]
                p_col = era5_1h_ds_col.pfull.values[::-1]
                z_col = era5_1h_ds_col.z.values[::-1]
                if np.any(vmr_h2o_col < 0):
                    print(
                        'ValueError: Found negative VMR value '
                        'in ERA5 profile. Continuing to next profile...',
                        flush=True)
                    continue
                if np.all(np.isnan(z_col)):
                    print(
                        'ValueError: Found only nan values in z profile. '
                        'Continuing to the next profile...', 
                        flush=True)
                    continue
                vmr_ref = ml.reference_h2o_vmr_profile(
                    vmr_h2o_col, p_col, z_col, from_mixed_layer_top=True)
                era5_1h_ds['q_ref'][:, ilat, ilon] = \
                    vmr2specific_humidity(vmr_ref)[::-1]
                eml_chars_col = ml.eml_characteristics(
                    vmr_h2o_col, vmr_ref, t_col, p_col, z_col,
                    min_eml_p_width=5000.,
                    min_eml_strength=0.2,
                    p_min=20000.,
                    p_max=90000.,
                    z_in_km=False,
                    p_in_hPa=False,
                    lat=lat,
                    lon=lon,
                    time=time.values,
                    )
                if np.isnan(eml_chars_col.strength[0]):
                    continue
                # vertical gridding of emls
                gridded_eml_ds = eec.eml_char_ds_to_pgrid(
                    eml_chars_col.to_xr_dataset(), 
                    p_col[::-1],
                    vertical_dim='fulllevel') 
                for eml_var in ['strength', 'pwidth', 'pmax', 'pmin', 
                                #'zmean', 'zmax', 'zmin'
                                ]:
                    era5_1h_ds[f'eml_{eml_var}'][:, ilat, ilon] = \
                        gridded_eml_ds[f'{eml_var}'].values 
        # print("Add IWV to dataset", flush=True)
        era5_1h_ds = add_iwv_to_ds(
            era5_1h_ds, q_var='q', iwv_varname='iwv')
        # era5_1h_ds = add_iwv_to_ds(
        #     era5_1h_ds, q_var='q_ref', iwv_varname='iwv_ref')
        print(f'Storing data {str(era5_1h_ds.time.values.astype("datetime64[h]"))}', flush=True)
        era5_1h_ds.to_netcdf(
            '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
            f'{str(era5_1h_ds.time.values.astype("datetime64[Y]"))}/'
            f'era5_3h_30N-S_eml_tropics_{str(era5_1h_ds.time.values.astype("datetime64[h]"))}.nc')
        era5_1h_ds.close()
        # if era5_1m_ds is None:
        #     era5_1m_ds = era5_1h_ds.copy()
        # else:
        #     era5_1m_ds = xr.concat([era5_1m_ds, era5_1h_ds], dim='time')
    queue.put(era5_1h_ds)
             
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="timestamp",
                    default='2020-06')
    args = parser.parse_args()
    month_str = "%02d" % (np.datetime64(args.time).astype('O').month)
    year_str = "%02d" % (np.datetime64(args.time).astype('O').year)
    
    
    era5_paths_lnsp = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_152_3hr_remapped.nc')
    era5_paths_q = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_133_3hr_remapped.nc')
    era5_paths_t = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_130_3hr_remapped.nc')
    era5_paths_u = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_131_3hr_remapped.nc')
    era5_paths_v = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_132_3hr_remapped.nc')
    era5_paths_w = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_135_3hr_remapped.nc')
    era5_paths_rr = \
        glob.glob('/scratch/u/u300676/data/era5_tropics/'
        f'era5_{year_str}-{month_str}_228_3hr_remapped.nc')

    era5_lnsp = xr.open_mfdataset(
        era5_paths_lnsp, combine='by_coords'
        )
    era5_q = xr.open_mfdataset(
        era5_paths_q, combine='by_coords'
        )
    era5_t = xr.open_mfdataset(
        era5_paths_t, combine='by_coords'
        )
    era5_u = xr.open_mfdataset(
        era5_paths_u, combine='by_coords'
        )
    era5_v = xr.open_mfdataset(
        era5_paths_v, combine='by_coords'
        )
    era5_w = xr.open_mfdataset(
        era5_paths_w, combine='by_coords'
        )
    era5_rr = xr.open_mfdataset(
        era5_paths_rr, combine='by_coords'
        )
    store_eml_ds_multi_proc(era5_lnsp, era5_q, era5_t, era5_u, era5_v, era5_w, era5_rr, args.time)
    
    # times = era5_lnsp.sel(
    #     time=args.time).time
    # era5_1m_ds = None
    # for time in times:
    #     print(f'calc eml characteristics for \n{time.values}', flush=True)
    #     lnsp = era5_lnsp.sel(time=time).lnsp.load()
    #     q = era5_q.sel(time=time).q.load()
    #     t = era5_t.sel(time=time).t.load()
    #     u = era5_u.sel(time=time).u.load()
    #     v = era5_v.sel(time=time).v.load()
    #     w = era5_w.sel(time=time).w.load()
    #     rr = era5_rr.sel(time=time).var228.load()

    #     era5_1h_ds = prepare_era5_ds(lnsp, q, t, u=u, v=v, w=w, rr=rr)
    #     era5_1h_ds = era5_1h_ds.assign(
    #         {
    #             f'{eml_var}': (('fulllevel', 'lat', 'lon'), 
    #                             np.full(era5_1h_ds.q.shape, np.nan))
    #             for eml_var in 
    #             ['eml_strength', 'eml_pmean', 'eml_pwidth', 
    #             'eml_pmax', 'eml_pmin', #'eml_zmean', 'eml_zmax', 'eml_zmin', 
    #             'q_ref']
    #         }
    #     )
    #     for ilat, lat in enumerate(era5_1h_ds.lat.values):
    #         print('calc eml characteristics for\n'
    #             f'lat {lat}\n', flush=True)
    #         for ilon, lon in enumerate(era5_1h_ds.pfull.lon.values):
    #             era5_1h_ds_col = era5_1h_ds.sel(
    #                 {'lat': lat, 'lon': lon}) 
    #             vmr_h2o_col = era5_1h_ds_col.vmr_h2o.values[::-1]
    #             t_col = era5_1h_ds_col.t.values[::-1]
    #             p_col = era5_1h_ds_col.pfull.values[::-1]
    #             z_col = era5_1h_ds_col.z.values[::-1]
    #             if np.any(vmr_h2o_col < 0):
    #                 print(
    #                     'ValueError: Found negative VMR value '
    #                     'in ERA5 profile. Continuing to next profile...',
    #                     flush=True)
    #                 continue
    #             if np.all(np.isnan(z_col)):
    #                 print(
    #                     'ValueError: Found only nan values in z profile. '
    #                     'Continuing to the next profile...', 
    #                     flush=True)
    #                 continue
    #             vmr_ref = ml.reference_h2o_vmr_profile(
    #                 vmr_h2o_col, p_col, z_col, from_mixed_layer_top=True)
    #             era5_1h_ds['q_ref'][:, ilat, ilon] = \
    #                 vmr2specific_humidity(vmr_ref)[::-1]
    #             eml_chars_col = ml.eml_characteristics(
    #                 vmr_h2o_col, vmr_ref, t_col, p_col, z_col,
    #                 min_eml_p_width=5000.,
    #                 min_eml_strength=0.2,
    #                 p_min=20000.,
    #                 p_max=90000.,
    #                 z_in_km=False,
    #                 p_in_hPa=False,
    #                 lat=lat,
    #                 lon=lon,
    #                 time=time.values,
    #                 )
    #             if np.isnan(eml_chars_col.strength[0]):
    #                 continue
    #             # vertical gridding of emls
    #             gridded_eml_ds = eec.eml_char_ds_to_pgrid(
    #                 eml_chars_col.to_xr_dataset(), 
    #                 p_col[::-1],
    #                 vertical_dim='fulllevel') 
    #             for eml_var in ['strength', 'pwidth', 'pmax', 'pmin', 
    #                             #'zmean', 'zmax', 'zmin'
    #                             ]:
    #                 era5_1h_ds[f'eml_{eml_var}'][:, ilat, ilon] = \
    #                     gridded_eml_ds[f'{eml_var}'].values 
    #     print("Add IWV to dataset", flush=True)
    #     era5_1h_ds = add_iwv_to_ds(
    #         era5_1h_ds, q_var='q', iwv_varname='iwv')
    #     era5_1h_ds = add_iwv_to_ds(
    #         era5_1h_ds, q_var='q_ref', iwv_varname='iwv_ref')
    #     if era5_1m_ds is None:
    #         era5_1m_ds = era5_1h_ds.copy()
    #     else:
    #         era5_1m_ds = xr.concat([era5_1m_ds, era5_1h_ds], dim='time')
    # print("Storing data", flush=True)
    # era5_1h_ds.to_netcdf(
    #     '/home/u/u300676/user_data/mprange/era5_tropics/eml_data/'
    #     f'era5_3h_30N-S_eml_tropics_{args.time}.nc')

if __name__ == '__main__':
    main()