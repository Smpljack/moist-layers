import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pickle
import pandas as pd
from typhon.math import integrate_column


def eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                        heating_rate=None,
                        ref_heating_rate=None,
                        w_rad=None,
                        w_rad_smooth=None,
                        w_rad_smooth_length=None,
                        min_eml_p_width=5000.,
                        min_eml_strength=1e-9,
                        p_min=10000.,
                        p_max=90000.,
                        z_in_km=False,
                        p_in_hPa=False,
                        lat=None,
                        lon=None,
                        time=None,
                        ):
    """
    Calculate moist layer characteristics for given H2O VMR profile.
    Moist layer characteristics are returned as MoistureCharacteristics
    objects.
    """
    z = select_pressure_range(z, p, p_min, p_max)
    t = select_pressure_range(t, p, p_min, p_max)
    h2o_vmr = select_pressure_range(h2o_vmr, p, p_min, p_max)
    ref_h2o_vmr = select_pressure_range(ref_h2o_vmr, p, p_min, p_max)
    p = select_pressure_range(p, p, p_min, p_max)
    theta = potential_temperature(t, p)
    dtheta_dp = np.diff(theta) / np.diff(p)

    anomaly = h2o_vmr - ref_h2o_vmr
    bound_ind = get_eml_bound_ind(anomaly)
    if bound_ind is None:
        return MoistureCharacteristics()
    eml_pressure_widths = p[bound_ind[:, 0]] - p[bound_ind[:, 1]]
    if not np.any(eml_pressure_widths > min_eml_p_width):
        return MoistureCharacteristics()
    bound_ind = bound_ind[eml_pressure_widths > min_eml_p_width, :]
    eml_pressure_widths = eml_pressure_widths[
        eml_pressure_widths > min_eml_p_width]
    eml_height_widths = z[bound_ind[:, 1]] - z[bound_ind[:, 0]]
    eml_strengths = np.array(
        [
            integrate_column(
                y=(anomaly)[bound_ind[ieml, 0]:bound_ind[ieml, 1]],
                x=z[bound_ind[ieml, 0]:bound_ind[ieml, 1]])
            for ieml in range(bound_ind.shape[0])
        ]
    ) / eml_height_widths
    if not np.any(eml_strengths > min_eml_strength):
        return MoistureCharacteristics()
    bound_ind = bound_ind[eml_strengths > min_eml_strength, :]
    eml_pressure_widths = eml_pressure_widths[
        eml_strengths > min_eml_strength]
    eml_height_widths = eml_height_widths[eml_strengths > min_eml_strength]
    eml_strengths = eml_strengths[eml_strengths > min_eml_strength]
    eml_inds = [np.arange(start, stop) for start, stop in bound_ind]
    anomaly_p_means = np.array(
        [anomaly_position(p[eml_ind], anomaly[eml_ind]) 
        for eml_ind in eml_inds]
    )
    anomaly_z_means = np.array(
        [anomaly_position(z[eml_ind], anomaly[eml_ind]) 
        for eml_ind in eml_inds]
    )
    anomaly_t_means = np.array(
        [anomaly_position(t[eml_ind], anomaly[eml_ind]) 
        for eml_ind in eml_inds]
    )
    anomaly_dtheta_dp_median = np.array(
        [np.median(dtheta_dp[eml_ind]) for eml_ind in eml_inds]
    ) 
    if heating_rate is not None:
        heating_rate = select_pressure_range(heating_rate, p, p_min, p_max)
        heating_rate_median = np.array(
            [np.median(
                heating_rate[bound_ind[ieml, 0]:bound_ind[ieml, 1]]) 
            for ieml in range(bound_ind.shape[0])]
        )
        heating_rate_min = np.array(
            [np.min(heating_rate[bound_ind[ieml, 0]:bound_ind[ieml, 1]]) 
            for ieml in range(bound_ind.shape[0])]
        )
        heating_rate_max = np.array(
            [np.max(heating_rate[bound_ind[ieml, 0]:bound_ind[ieml, 1]]) 
            for ieml in range(bound_ind.shape[0])]
        )
        heating_rate_10p = np.array(
            [np.percentile(
                heating_rate[bound_ind[ieml, 0]:bound_ind[ieml, 1]], 10) 
                for ieml in range(bound_ind.shape[0])]
        ) 
        heating_rate_90p = np.array(
            [np.percentile(
                heating_rate[bound_ind[ieml, 0]:bound_ind[ieml, 1]], 90) 
                for ieml in range(bound_ind.shape[0])]
        ) 
        if ref_heating_rate is not None:
            ref_heating_rate = select_pressure_range(
                ref_heating_rate, p, p_min, p_max)
            heating_rate_anom_means = np.array(
                [np.mean(
                    (ref_heating_rate - heating_rate)
                    [bound_ind[ieml, 0]:bound_ind[ieml, 1]]) 
                    for ieml in range(bound_ind.shape[0])]
            )
            heating_rate_anom_min = np.array(
                [np.min(
                    (ref_heating_rate - heating_rate)
                    [bound_ind[ieml, 0]:bound_ind[ieml, 1]]) 
                    for ieml in range(bound_ind.shape[0])]
            )
            heating_rate_anom_max = np.array(
                [np.max((ref_heating_rate-heating_rate)
                [bound_ind[ieml, 0]:bound_ind[ieml, 1]]) 
                for ieml in range(bound_ind.shape[0])]
        )
        else: 
                heating_rate_anom_means = np.nan * np.ones(
                    bound_ind.shape[0])
                heating_rate_anom_min = np.nan * np.ones(
                    bound_ind.shape[0])
                heating_rate_anom_max = np.nan * np.ones(
                    bound_ind.shape[0])
                heating_rate_anom_10p = np.nan * np.ones(
                    bound_ind.shape[0])
                heating_rate_anom_90p = np.nan * np.ones(
                    bound_ind.shape[0])
    else:
        heating_rate_median = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_min = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_max = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_10p = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_90p = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_anom_means = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_anom_min = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_anom_max = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_anom_10p = np.nan * np.ones(bound_ind.shape[0])
        heating_rate_anom_90p = np.nan * np.ones(bound_ind.shape[0])
    if w_rad is not None:
        w_rad = select_pressure_range(w_rad, p, p_min, p_max)
        if w_rad_smooth_length is not None:
            w_rad_smooth = smooth_wrad(w_rad, z, w_rad_smooth_length)
            wrad_median = np.array(
                [np.median(
                    w_rad_smooth[bound_ind[ieml, 0]:bound_ind[ieml, 1]])
    for ieml in range(bound_ind.shape[0])])
        else:
            wrad_median = np.array(
            [np.median(w_rad[bound_ind[ieml, 0]:bound_ind[ieml, 1]])
                for ieml in range(bound_ind.shape[0])])
    else:
        wrad_median = np.nan * np.ones(bound_ind.shape[0])
    if w_rad_smooth is not None:
        wrad_smooth_median = np.array(
            [np.median(w_rad_smooth[bound_ind[ieml, 0]:bound_ind[ieml, 1]])
                for ieml in range(bound_ind.shape[0])]) 
    else:
        wrad_smooth_median = np.nan * np.ones(bound_ind.shape[0])
        
    characteristics = MoistureCharacteristics(
        pmin=p[bound_ind[:, 1]],
        pmax=p[bound_ind[:, 0]],
        zmin=z[bound_ind[:, 0]],
        zmax=z[bound_ind[:, 1]],
        strength=eml_strengths,
        pwidth=eml_pressure_widths,
        zwidth=eml_height_widths,
        pmean=anomaly_p_means,
        zmean=anomaly_z_means,
        tmean=anomaly_t_means, 
        tmin=t[bound_ind[:, 1]],
        tmax=t[bound_ind[:, 1]], 
        heating_rate_med=heating_rate_median,
        heating_rate_min=heating_rate_min,
        heating_rate_max=heating_rate_max,
        heating_rate_10p=heating_rate_10p,
        heating_rate_90p=heating_rate_90p,
        heating_rate_anomaly_mean=heating_rate_anom_means,
        heating_rate_anomaly_min=heating_rate_anom_min,
        heating_rate_anomaly_max=heating_rate_anom_max,
        wrad_median=wrad_median,
        wrad_smooth_median=wrad_smooth_median,
        dtheta_dp_median=anomaly_dtheta_dp_median,
        lat=lat,
        lon=lon,
        time=time,
    )
    if z_in_km:
        characteristics.to_km()
    if p_in_hPa:
        characteristics.to_hpa()
    return characteristics

def smooth_wrad(w_rad, z, w_rad_smooth_length):
    """
    Smooth radiatively driven vertical pressure velocity.
    """
    wrad_df = pd.DataFrame(data={'wrad': w_rad}, index=z)
    z_interp = np.arange(z.min(), z.max(), 10)
    wrad_interp_df = interp(wrad_df, z_interp)
    w_rad_smooth_interp = wrad_interp_df.rolling(
        window=int(w_rad_smooth_length / 10), 
        min_periods=1).mean()
    w_rad_smooth = interp(w_rad_smooth_interp, z).values[:, 0]
    return w_rad_smooth

def smooth_in_height(data, z, smooth_length, interp_interval=10):
    """
    Smooth data in height coordinate.
    """
    df = pd.DataFrame(data={'data': data}, index=z)
    z_interp = np.arange(z.min(), z.max(), interp_interval)
    interp_df = interp(df, z_interp)
    smooth_interp = interp_df.rolling(
        window=int(smooth_length / interp_interval), 
        min_periods=1).mean()
    smooth = interp(smooth_interp, z).values[:, 0]
    return smooth


def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out

def anomaly_position(grid, anomaly):
    """
    Calculate anomaly position. Regridding assures equally spaced grid, 
    important for getting correct average in the end.
    """
    anomaly_f = interp1d(
        grid,
        anomaly,
    )
    new_grid = np.linspace(
        grid.min(),
        grid.max(),
        num=len(grid)*1000,
    )
    if np.diff(grid)[0] < 0:
        new_grid = new_grid[::-1]
    anomaly_regrid = anomaly_f(new_grid)
    anomaly_position = np.average(new_grid, weights=anomaly_regrid)
    return anomaly_position

def reference_h2o_vmr_profile(h2o_vmr, p, z, p_min=10000., p_max=None):
    """
    Calculate reference H2O VMR profile as a logarithmic 
    quadratic fit against the H2O VMR profile of interest.
    """
    is_tropo = p > p_min
    if p_max is not None:
        is_tropo = p < p_max
    popt, _ = curve_fit(
        ref_profile_opt_func,
        z[is_tropo],
        np.log(h2o_vmr[is_tropo]),
        bounds=([-np.inf, -np.inf,
                 np.log(h2o_vmr[is_tropo][0] - 1e-9)],
                [np.inf, np.inf,
                 np.log(h2o_vmr[is_tropo][0] + 1e-9)]),
        # check_finite=False,
    )
    ref_profile = np.exp(np.polyval(popt, z))

    return ref_profile
    
def ref_profile_opt_func(z, a, b, c):
    """
    Function used for fitting the reference humidity profiles.
    A quadratic function is used and applied to logarithmic H2O VMR profiles.
    """
    return a * z**2 + b * z + c
    
def select_pressure_range(data, p, p_min, p_max):
    """
    Select data within pressure range p_min, p_max.
    """
    p_ind = p > p_min
    p_ind &= p < p_max
    return data[p_ind]

def get_eml_bound_ind(anomaly):
    """
    Get indices of vertical bounds of positive moisture anomalies, 
    i.e. of moist layers.
    """
    moist = anomaly > 0
    eml_bound_ind = (np.nonzero(moist[1:] != moist[:-1])[0] + 1)
    if moist[0]: # Add limit at index 0, if there is an anomaly
        eml_bound_ind = np.concatenate([[0], eml_bound_ind])
    if moist[-1]: # Add limit at the index -1, if there is an anomaly
        eml_bound_ind = np.concatenate([eml_bound_ind, [len(moist) - 1]])
    eml_bound_ind = eml_bound_ind.reshape(-1, 2)
    if eml_bound_ind.shape[0] == 0:
        return None
    # Drop anomaly, if it's bound by upper pressure (lower z limit)
    if eml_bound_ind[0, 0] == 0: 
        eml_bound_ind = np.delete(eml_bound_ind, axis=0, obj=[0])
    if eml_bound_ind.shape[0] == 0:
        return None
    # Drop anomaly, if it's bound by lower pressure (upper z limit)
    if eml_bound_ind[-1, -1] == len(moist) - 1: 
        eml_bound_ind = np.delete(eml_bound_ind, axis=0, obj=[-1])
    return eml_bound_ind

class MoistureCharacteristics:
    """
    Represents vertical moist layer characteristics of a humidity profile.
    Contains characteristics of positive moisture anomalies within the 
    humidity profile, like moist layer height, strength and thickness.
    """
    def __init__(self,
                 pmin=np.array([np.nan]),
                 pmax=np.array([np.nan]),
                 zmin=np.array([np.nan]),
                 zmax=np.array([np.nan]),
                 strength=np.array([np.nan]),
                 pwidth=np.array([np.nan]),
                 zwidth=np.array([np.nan]),
                 zmean=np.array([np.nan]),
                 pmean=np.array([np.nan]),
                 tmean=np.array([np.nan]),
                 tmin=np.array([np.nan]),
                 tmax=np.array([np.nan]),
                 heating_rate_med=np.array([np.nan]),
                 heating_rate_min=np.array([np.nan]),
                 heating_rate_max=np.array([np.nan]),
                 heating_rate_10p=np.array([np.nan]),
                 heating_rate_90p=np.array([np.nan]),
                 heating_rate_anomaly_mean=np.array([np.nan]),
                 heating_rate_anomaly_min=np.array([np.nan]),
                 heating_rate_anomaly_max=np.array([np.nan]),
                 wrad_median=np.array([np.nan]),
                 wrad_smooth_median=np.array([np.nan]),
                 dtheta_dp_median=np.array([np.nan]),
                 lat=np.array([np.nan]),
                 lon=np.array([np.nan]),
                 time=np.array([np.nan]),
                 ):
        self.pmin = pmin
        self.pmax = pmax
        self.zmin = zmin
        self.zmax = zmax
        self.strength = strength
        self.pwidth = pwidth
        self.zwidth = zwidth
        self.pmean = pmean
        self.zmean = zmean
        self.tmean = tmean
        self.tmin = tmin
        self.tmax = tmax
        self.heating_rate_med = heating_rate_med
        self.heating_rate_min = heating_rate_min
        self.heating_rate_max = heating_rate_max
        self.heating_rate_10p = heating_rate_10p
        self.heating_rate_90p = heating_rate_90p
        self.heating_rate_anomaly_mean = heating_rate_anomaly_mean
        self.heating_rate_anomaly_min = heating_rate_anomaly_min
        self.heating_rate_anomaly_max = heating_rate_anomaly_max
        self.wrad_median = wrad_median
        self.wrad_smooth_median = wrad_smooth_median
        self.dtheta_dp_median = dtheta_dp_median
        self.lat = lat
        self.lon = lon
        self.time = time

    def __getitem__(self, index):
        return self.__init__(
                  pmin=np.array([self.pmin[index]]).reshape(len(index)),
                  pmax=np.array([self.pmax[index]]).reshape(len(index)),
                  zmin=np.array([self.zmin[index]]).reshape(len(index)),
                  zmax=np.array([self.zmax[index]]).reshape(len(index)),
                  strength=np.array([self.strength[index]]).reshape(len(index)),
                  pwidth=np.array([self.pwidth[index]]).reshape(len(index)),
                  zwidth=np.array([self.zwidth[index]]).reshape(len(index)),
                  zmean=np.array([self.zmean[index]]).reshape(len(index)),
                  pmean=np.array([self.pmean[index]]).reshape(len(index)),
                  tmean=np.array([self.tmean[index]]).reshape(len(index)),
                  tmin=np.array([self.tmin[index]]).reshape(len(index)),
                  tmax=np.array([self.tmax[index]]).reshape(len(index)),
                  heating_rate_med=np.array([self.heating_rate_med[index]]).reshape(len(index)),
                  heating_rate_min=np.array([self.heating_rate_min[index]]).reshape(len(index)),
                  heating_rate_max=np.array([self.heating_rate_max[index]]).reshape(len(index)),
                  heating_rate_10p=np.array([self.heating_rate_10p[index]]).reshape(len(index)),
                  heating_rate_90p=np.array([self.heating_rate_90p[index]]).reshape(len(index)),
                  heating_rate_anomaly_mean=np.array([self.heating_rate_anomaly_mean[index]]).reshape(len(index)),
                  heating_rate_anomaly_min=np.array([self.heating_rate_anomaly_min[index]]).reshape(len(index)),
                  heating_rate_anomaly_max=np.array([self.heating_rate_anomaly_max[index]]).reshape(len(index)),
                  wrad_median=np.array([self.wrad_median[index]]).reshape(len(index)),
                  wrad_smooth_median=np.array([self.wrad_median[index]]).reshape(len(index)),
                  dtheta_dp_median=np.array([self.dtheta_dp_median[index]]).reshape(len(index)),
        )

    def to_hpa(self):
        self.pmin /= 100.
        self.pmax /= 100.
        self.pwidth /= 100.
        self.pmean /= 100.

    def to_km(self):
        self.zmin /= 1000.
        self.zmax /= 1000.
        self.zwidth /= 1000.
        self.zmean /= 1000.

    def get_max_strength_anomaly(self):
        max_str_anomaly = self.__getitem__(index=self.strength.argmax())
        return max_str_anomaly
    
    def to_xr_dataset(self):
        n_eml = len(self.strength)
        d = xr.Dataset(
            coords={
                'eml_count': np.arange(n_eml),
            },
            data_vars={
                'strength': (('eml_count'), self.strength),
                'pmean': (('eml_count'), self.pmean),
                'pwidth': (('eml_count'), self.pwidth),
                'zwidth': (('eml_count'), self.zwidth),
                'lat': (('eml_count'), self.lat.repeat(n_eml)),
                'lon': (('eml_count'), self.lon.repeat(n_eml)),
                'time': (('eml_count'), self.time.repeat(n_eml)),
            }
        )
        return d


def save_object_to_pickle(obj, filename):
    """Store python object as pickle file."""
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_pickle(filename):
    """Load pickle file."""
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def potential_temperature(t, p):
    """Calculate potential temperature."""
    return t * (100000/p)**0.286

def wtg(T,Tpot,Qcool_day,dTpot_dp):
    """ Calculate vertical velocity by using 
    the weak temperature gradient approximation 
    Input:
    Tpot:  potential temperature
    Qcool: cooling [K/d]
    Output:
    vertical velocity in Pa/s
    omega = Qcool/S where S= T/Tpot * dTpot/dp
    """
    Qcool_sec = Qcool_day/(24*60*60.)
    return Qcool_sec/(T/Tpot*(dTpot_dp))


def mask_eurec4a(ds, grid):
    """
    Return EUREC4A domain.
    """
    mask_eurec4a = (
        (grid.clon > np.deg2rad(-65)) &
        (grid.clon < np.deg2rad(-40)) &
        (grid.clat > np.deg2rad(5)) &
        (grid.clat < np.deg2rad(25))
    )
    return ds.isel(cell=mask_eurec4a)

def mask_cross_section(ds, grid):
    mask_cs = (
        (grid.clon > np.deg2rad(-45.5)) &
        (grid.clon < np.deg2rad(-44.5)) &
        (grid.clat > np.deg2rad(8)) &
        (grid.clat < np.deg2rad(25))
    )
    return ds.isel(cell=mask_cs)

def mask_point(ds, grid):
    mask_point = (
    (grid.clon > np.deg2rad(-45.5)) &
    (grid.clon < np.deg2rad(-44.5)) &
    (grid.clat > np.deg2rad(21.5)) &
    (grid.clat < np.deg2rad(22.5))
    )
    return ds.isel(cell=mask_point)
