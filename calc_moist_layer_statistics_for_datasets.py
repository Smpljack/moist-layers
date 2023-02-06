import numpy as np

from typhon.physics import specific_humidity2vmr, pressure2height
from moist_layers import (eml_characteristics, reference_h2o_vmr_profile, 
                          heating_rate, potential_temperature, wtg)


def calc_eml_characteristics_airs_era5(airs_era5_colloc, **kwargs):
    """
    Calculate moist layer characteristics for collocations between 
    the AIRS CLIMCAPS L2 retrieval and ERA5.
    """
    airs_chars = []
    era5_chars = []
    for i in range(len(airs_era5_colloc['airs/time_rand'].values)):
        # AIRS
        na = np.isnan(airs_era5_colloc['airs/vmr_rand'][i, :].values)
        h2o_vmr = airs_era5_colloc['airs/vmr_rand'][i, ~na].values[::-1]
        h2o_vmr_surface = specific_humidity2vmr(
            airs_era5_colloc['airs/q_surf_rand'][i].values)
        h2o_vmr = np.concatenate([[h2o_vmr_surface], h2o_vmr])
        p = airs_era5_colloc['airs/air_pres_h2o'][~na].values[::-1]
        ps = airs_era5_colloc['airs/prior_surf_pres_rand'][i].values 
        p = np.concatenate([[ps], p])
        t = airs_era5_colloc['airs/ta_rand_h2o_grid'][i, ::-1].values[~na]
        ts = airs_era5_colloc['airs/surface_temperature_rand'][i].values
        t = np.concatenate([[ts], t])
        Q = airs_era5_colloc['airs/lw_heating_rate_rand'][i, ::-1].values[~na] 
        Q = np.concatenate([[0], Q])
        z = airs_era5_colloc['airs/gp_height_h2o'][i, ~na].values[::-1]
        z = np.concatenate([[0], z])
        
        w_rad = airs_era5_colloc['airs/lw_wrad_rand'][i, ::-1].values[~na]
        w_rad = np.concatenate([[0], w_rad])
        w_rad_smooth = airs_era5_colloc[
            'airs/lw_wrad_rand_smooth'][i, ::-1].values[~na]
        w_rad_smooth = np.concatenate([[0], w_rad_smooth])
        
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of AIRS reference profile {i}")
            airs_chars.append(np.nan)
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of AIRS reference profile {i}")
            airs_chars.append(np.nan)
        else:
            char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z, 
                                    heating_rate=Q, w_rad=w_rad, 
                                    w_rad_smooth=w_rad_smooth, **kwargs)
            airs_chars.append(char)
        #ERA5
        h2o_vmr = airs_era5_colloc['era5/vmr_h2o'][i, ::-1].values
        p = airs_era5_colloc['era5/p'][i, ::-1].values
        t = airs_era5_colloc['era5/t'][i, ::-1].values
        z = airs_era5_colloc['era5/z'][i, ::-1].values
        Q = airs_era5_colloc['era5/lw_heating_rate'][i, ::-1].values
        w_rad = airs_era5_colloc['era5/lw_wrad'][i, ::-1].values
        w_rad_smooth = airs_era5_colloc['era5/lw_wrad_smooth'][i, ::-1].values
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of ERA5 reference profile {i}")
            era5_chars.append(np.nan)
            airs_chars[-1] = np.nan
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of ERA5 reference profile {i}")
            era5_chars.append(np.nan)
            airs_chars[-1] = np.nan
        else:
            if type(airs_chars[-1]) is float:
                era5_chars.append(np.nan)
            char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                       heating_rate=Q, w_rad=w_rad, 
                                       w_rad_smooth=w_rad_smooth, **kwargs)
            era5_chars.append(char)
        
    return {'airs_chars': airs_chars, 'era5_chars': era5_chars}

def calc_eml_characteristics_era5_gruan(era5_gruan_colloc, **kwargs):
    """
    Calculate moist layer characteristics for collocations between 
    ERA5 and GRUAN radiosonde data.
    """
    era5_chars = []
    rs_chars = []
    for i in range(len(era5_gruan_colloc['time'].values)):
        # ERA5
        h2o_vmr = era5_gruan_colloc['era5/vmr_h2o_rand'][i, ::-1].values
        p = era5_gruan_colloc['era5/p_rand'][i, ::-1].values
        t = era5_gruan_colloc['era5/t_rand'][i, ::-1].values
        z = era5_gruan_colloc['era5/z_rand'][i, ::-1].values
        Q = era5_gruan_colloc['era5/lw_heating_rate_rand'][i, ::-1].values
        w_rad = era5_gruan_colloc['era5/lw_wrad_rand'][i, ::-1].values
        w_rad_smooth = era5_gruan_colloc[
            'era5/lw_wrad_rand_smooth'][i, ::-1].values

        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of ERA5 reference profile {i}")
            era5_chars.append(np.nan)
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of ERA5 reference profile {i}")
            era5_chars.append(np.nan)
        else:
            char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                    heating_rate=Q, w_rad=w_rad, 
                                    w_rad_smooth=w_rad_smooth, **kwargs)
            era5_chars.append(char)
        # GRUAN
        na = np.isnan(era5_gruan_colloc['radiosonde/vmr'][i, :])
        na &= np.isnan(era5_gruan_colloc['radiosonde/ta'][i, :])
        h2o_vmr = era5_gruan_colloc['radiosonde/vmr'][i, :].values[~na]
        p = era5_gruan_colloc['radiosonde/p'][i, :].values[~na]
        t = era5_gruan_colloc['radiosonde/ta'][i, :].values[~na]
        Q = era5_gruan_colloc['radiosonde/lw_heating_rate'][i, :].values[~na] 
        z = era5_gruan_colloc['radiosonde/height'][:].values[~na]
        w_rad = era5_gruan_colloc['radiosonde/lw_wrad'][i, :].values[~na] 
        w_rad_smooth = era5_gruan_colloc[
            'radiosonde/lw_wrad_smooth'][i, :].values[~na] 
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of GRUAN reference profile {i}")
            rs_chars.append(np.nan)
            era5_chars[-1] = np.nan
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of GRUAN reference profile {i}")
            rs_chars.append(np.nan)
            era5_chars[-1] = np.nan
        else:
            if type(era5_chars[-1]) is float:
                rs_chars.append(np.nan)
            else:
                char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                           heating_rate=Q, w_rad=w_rad, 
                                           w_rad_smooth=w_rad_smooth, **kwargs)
                rs_chars.append(char)
    return {'rs_chars': rs_chars, 'era5_chars': era5_chars}
    
def calc_eml_characteristics_iasi_reprocessed_gruan(iasi_gruan_colloc, 
                                                    **kwargs):
    """
    Calculate moist layer characteristics for collocations between 
    the IASI L2 Climate Data Record retrieval and GRUAN radiosonde data.
    """
    rs_chars = []
    iasi_chars = []
    for i in range(len(iasi_gruan_colloc['collocation'].values)):
        # RADIOSONDE
        na = np.isnan(iasi_gruan_colloc['radiosonde/vmr'][i, :])
        na &= np.isnan(iasi_gruan_colloc['radiosonde/ta'][i, :])
        h2o_vmr = iasi_gruan_colloc['radiosonde/vmr'][i, :].values[~na]
        p = iasi_gruan_colloc['radiosonde/p'][i, :].values[~na]
        t = iasi_gruan_colloc['radiosonde/ta'][i, :].values[~na]
        Q = iasi_gruan_colloc['radiosonde/lw_heating_rate'][i, :].values[~na] 
        z = iasi_gruan_colloc['radiosonde/height'][:].values[~na]
        w_rad = iasi_gruan_colloc['radiosonde/lw_wrad'][i, :].values[~na] 
        w_rad_smooth = iasi_gruan_colloc[
            'radiosonde/lw_wrad_smooth'][i, :].values[~na] 
        
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of reference profile {i}")
            rs_chars.append(np.nan)
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of reference profile {i}")
            rs_chars.append(np.nan)
        else:
            char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                        heating_rate=Q, w_rad=w_rad,
                                        w_rad_smooth=w_rad_smooth, **kwargs)
            rs_chars.append(char)
        # IASI
        na = np.isnan(iasi_gruan_colloc['iasi/vmr_rand'][i, :].values)
        na &= np.isnan(iasi_gruan_colloc['iasi/ta_rand'][i, :].values)
        h2o_vmr = iasi_gruan_colloc['iasi/vmr_rand'][i, ~na].values[::-1]
        p = iasi_gruan_colloc[
            'iasi/pressure_levels_rand'][i, ~na].values[::-1] * 100
        t = iasi_gruan_colloc['iasi/ta_rand'][i, ~na].values[::-1]
        Q = iasi_gruan_colloc['iasi/lw_heating_rate_rand'][i, ~na].values[::-1]
        z = pressure2height(p, t)
        w_rad = iasi_gruan_colloc['iasi/lw_wrad_rand'][i, ~na].values[::-1]
        w_rad_smooth = iasi_gruan_colloc[
            'iasi/lw_wrad_rand_smooth'][i, ~na].values[::-1]
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                    f"creation of reference profile {i}")
            iasi_chars.append(np.nan)
            rs_chars[-1] = np.nan
        except IndexError:
            print("IndexError: Encountered IndexError during "
                    f"creation of reference profile {i}")
            iasi_chars.append(np.nan)
            rs_chars[-1] = np.nan
        else:
            if type(rs_chars[-1]) is float:
                iasi_chars.append(np.nan)
            else:
                char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                            heating_rate=Q, w_rad=w_rad,
                                            w_rad_smooth=w_rad_smooth, **kwargs)
                iasi_chars.append(char)
       
    return {'iasi_chars': iasi_chars, 'rs_chars': rs_chars}

    
def calc_eml_characteristics_iasi_reprocessed_era5(iasi_era5_colloc, **kwargs):
    """
    Calculate moist layer characteristics for collocations between 
    the IASI L2 Climate Data Record retrieval and ERA5.
    """
    iasi_chars = []
    era5_chars = []
    for i in range(len(iasi_era5_colloc['iasi/time_rand'].values)):
        na = np.isnan(iasi_era5_colloc['iasi/vmr_rand'][i, :].values)
        na &= np.isnan(iasi_era5_colloc['iasi/ta_rand'][i, :].values)
        h2o_vmr = iasi_era5_colloc['iasi/vmr_rand'][i, ~na].values[::-1]
        p = iasi_era5_colloc['iasi/pressure_levels_rand'][i, ~na].values[::-1] * 100
        t = iasi_era5_colloc['iasi/ta_rand'][i, ~na].values[::-1]
        Q = iasi_era5_colloc['iasi/lw_heating_rate_rand'][i, ~na].values[::-1]
        z = pressure2height(p, t)
        w_rad = iasi_era5_colloc['iasi/lw_wrad_rand'][i, ~na].values[::-1]
        w_rad_smooth = iasi_era5_colloc[
            'iasi/lw_wrad_rand_smooth'][i, ~na].values[::-1]
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of reference profile {i}")
            iasi_chars.append(np.nan)
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of reference profile {i}")
            iasi_chars.append(np.nan)
        else:
            char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                        heating_rate=Q, w_rad=w_rad, 
                                        w_rad_smooth=w_rad_smooth, **kwargs)
            iasi_chars.append(char)
        
        h2o_vmr = iasi_era5_colloc['era5/vmr_h2o'][i, ::-1].values
        p = iasi_era5_colloc['era5/p'][i, ::-1].values
        t = iasi_era5_colloc['era5/t'][i, ::-1].values
        z = iasi_era5_colloc['era5/z'][i, ::-1].values
        Q = iasi_era5_colloc['era5/lw_heating_rate'][i, ::-1].values
        w_rad = iasi_era5_colloc['era5/lw_wrad'][i, ::-1].values
        w_rad_smooth = iasi_era5_colloc['era5/lw_wrad_smooth'][i, ::-1].values
        try:
            ref_h2o_vmr = reference_h2o_vmr_profile(h2o_vmr, p, z)
        except ValueError:
            print("ValueError: Encountered ValueError during "
                f"creation of reference profile {i}")
            era5_chars.append(np.nan)
            iasi_chars[-1] = np.nan
        except IndexError:
            print("IndexError: Encountered IndexError during "
                f"creation of reference profile {i}")
            era5_chars.append(np.nan)
            iasi_chars[-1] = np.nan
        else:
            if type(iasi_chars[-1]) is float:
                era5_chars[-1] = np.nan
            else:
                char = eml_characteristics(h2o_vmr, ref_h2o_vmr, t, p, z,
                                            heating_rate=Q, w_rad=w_rad,
                                            w_rad_smooth=w_rad_smooth, 
                                            **kwargs)
                era5_chars.append(char)
    return {'iasi_chars': iasi_chars, 'era5_chars': era5_chars}

def smooth_gruan(gruan_colloc, z_interval):
    """
    Smooth GRUAN radiosonde data of a collocation dataset. Smoothing is applied
    to all GRUAN variables within the dataset.
    Heating rates are calculated again based on smoothed variables.
    """
    # Radiosonde data is evenly spaced on 10 m vertical grid.
    # Therefore, averaging window is z_interval/10.
    gruan_colloc['radiosonde/vmr'] = \
        gruan_colloc['radiosonde/vmr'].rolling(
            dim={'radiosonde/height': int(z_interval/10)},
            center=True, min_periods=1).mean()
    gruan_colloc['radiosonde/ta'] = \
        gruan_colloc['radiosonde/ta'].rolling(
            dim={'radiosonde/height': int(z_interval/10)},
            center=True, min_periods=1).mean()
    gruan_colloc = gruan_colloc.assign_attrs(
        {'smoothing_interval': str(z_interval) + ' meters'}
    )
    # Heating rates are on half levels
    Q = np.nan * np.ones(gruan_colloc['radiosonde/p'].shape) 
    # Heating rates are on half levels
    wrad = np.nan * np.ones(gruan_colloc['radiosonde/p'].shape) 
    for i in range(len(gruan_colloc['collocation'])):
        mask = ~np.isnan(gruan_colloc['radiosonde/p'][i, :].values)
        mask &= ~np.isnan(gruan_colloc['radiosonde/ta'][i, :].values)
        mask &= ~np.isnan(gruan_colloc['radiosonde/vmr'][i, :].values)
        p = gruan_colloc['radiosonde/p'][i, mask].values
        z = gruan_colloc['radiosonde/height'][mask].values
        t = gruan_colloc['radiosonde/ta'][i, mask].values
        h2o_vmr = gruan_colloc['radiosonde/vmr'][i, mask].values
        Q[i, mask] = heating_rate(p, t, h2o_vmr, kind='lw')
        theta = potential_temperature(t, p)
        dtheta_dp = np.diff(theta) / np.diff(p)
        mask_wrad = mask.copy()
        mask_wrad[np.where(mask_wrad)[0][0]] = False 
        wrad[i, mask_wrad] = wtg(t[1:], theta[1:], Q[i, mask_wrad], dtheta_dp)
        gruan_colloc = gruan_colloc.assign(
            {'radiosonde/lw_heating_rate': (('collocation', 'radiosonde/height'), Q)})
        gruan_colloc = gruan_colloc.assign(
            {'radiosonde/lw_wrad': (('collocation', 'radiosonde/height'), wrad)})
    return gruan_colloc
