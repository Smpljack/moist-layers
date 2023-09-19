import konrad
from scipy.interpolate import interp1d
import numpy as np
import glob
import xarray as xr
import argparse

from typhon.physics import specific_humidity2vmr

def plev2phlev(plev):
    """Convert full-level to half-level pressure"""
    f = interp1d(
        x=np.arange(plev.size),
        y=np.log(plev),
        fill_value="extrapolate",
    )
    return np.exp(f(np.arange(-0.5, plev.size)))


def heating_rate(p, t, h2o_vmr, kind='net'):
    ph = plev2phlev(p)
    # print("P SHAPE")
    # print(ph.shape)
    atm = konrad.atmosphere.Atmosphere(ph)
    atm['T'][:] = t
    atm['H2O'][:] = h2o_vmr
    p_add = 0 # Count added plevs.
    while ph.min() > 9000:
        p_add += 1
        ph = np.concatenate([ph, [ph.min() * 0.9]])
    atm = atm.refine_plev(ph)
    atm['H2O'][atm['H2O'] < 0] = 1e-9
    rrtmg = konrad.radiation.RRTMG()
    rrtmg.update_heatingrates(
        atm,
        surface=konrad.surface.SlabOcean.from_atmosphere(atm),
        cloud=konrad.cloud.ClearSky.from_atmosphere(atm)
    )
    if p_add > 0:
        return rrtmg[f'{kind}_htngrt'][0][:-p_add] # Only return original plevs.
    else:
        return rrtmg[f'{kind}_htngrt'][0]
        
def interp_nan(profile):
    n_na = np.isnan(profile).sum()
    print(f"Found {n_na} missing values in profile."
            "Interpolating...", flush=True)
    # In case of missing values, interpolate over them.
    x = np.arange(len(profile))
    profile = np.interp(x, x[~np.isnan(profile)], profile[~np.isnan(profile)])
    return profile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str,
                    help="time",
                    default="2021-07-28")
    args = parser.parse_args()
    # Load gridded EML data
    eml_ds_paths = sorted(
        glob.glob('/home/u/u300676/user_data/mprange/eml_data/gridded/'
                  'gridded_monsoon_0p25deg_eml_tropics_2021-*'))

    eml_ds = xr.open_mfdataset(
        eml_ds_paths, concat_dim='time', combine='nested').sel(
            time=args.time
        )
    
    eml_ds = eml_ds.assign(
        {
            'heating_rate_lw': ((eml_ds.hus.dims), 
                                np.full(eml_ds.hus.shape, np.nan))
        }
    ).sel(lon=slice(-179, 179))

    for itime, time in enumerate(eml_ds.time.values):
        print(f'Loading data for {time}', flush=True)
        eml_ds_time = eml_ds.sel(time=time).load()
        for ilat, lat in enumerate(eml_ds.lat.values):
            print(
                f"Calculating heating rate for\nlat/time\t{lat}/{time}\n",
                flush=True)
            for ilon, lon in enumerate(eml_ds.lon.values):
                eml_ds_col = eml_ds_time.isel(
                    {'lat': ilat, 'lon': ilon})

                p = eml_ds_col.pfull.values[::-1][:48]
                n_na = np.isnan(p).sum()
                if n_na > 0:
                    p = interp_nan(p)

                t = eml_ds_col.ta.values[::-1][:48]
                n_na = np.isnan(t).sum()
                if n_na > 0:
                    t = interp_nan(t)

                q = eml_ds_col.hus.values[::-1][:48]
                n_na = np.isnan(q).sum()
                if n_na > 0:
                    q = interp_nan(q)

                h2o_vmr = specific_humidity2vmr(q)
                eml_ds_time['heating_rate_lw'][42:, ilat, ilon] = \
                    heating_rate(p, t, h2o_vmr, kind='lw')[::-1]

        eml_ds_time.to_netcdf(
            '/home/u/u300676/user_data/mprange/eml_data/gridded/'
            f'hr_gridded_monsoon_0p25deg_eml_tropics_{str(time)[:13]}.nc')

if __name__ == '__main__':
    main()