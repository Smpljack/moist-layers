import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import eval_eml_chars as eec
import cartopy.crs as ccrs
from typhon.plots import worldmap
import global_land_mask as globe
from metpy.calc import static_stability
from metpy.units import units

def main():
    month = '07'
    # Load gridded EML data
    eml_ds_paths = np.sort(
        glob.glob(
                '/home/u/u300676/user_data/mprange/eml_data/gridded/monthly_means/geographical/'
                f"gridded_monsoon_0p25deg_eml_tropics_*_2021-{month}_geographical_mean.nc"
                ))
    eml_ds_paths = [p for p in eml_ds_paths 
                    if (('_rh_' in p) or
                        ('_ta_' in p) or
                        ('_ua_' in p) or
                        ('_pfull_' in p) or
                        ('_va_' in p))
                    ]
    eml_ds = xr.open_mfdataset(
        eml_ds_paths)
    eml_ds = eml_ds.sel(
                {
                    'lon': slice(-20, 0),
                    'lat': slice(0, 30),
                    'fulllevel': slice(43, 91),
                }
            )
    lat_mesh, lon_mesh = np.meshgrid(eml_ds.lat, eml_ds.lon) 
    is_ocean = xr.DataArray(
                globe.is_ocean(lat_mesh, lon_mesh), 
                dims=['lon', 'lat'])
    eml_ds = eml_ds.where(~is_ocean).mean('lon').dropna('lat', how='all')
    fig, axs = plt.subplots()
    pc = axs.pcolormesh(
        eml_ds.lat, eml_ds.pfull_mean/100, eml_ds.rh_mean*100,
        cmap='density', vmin=0, vmax=100)
    cb2 = plt.colorbar(
        pc, orientation="vertical", shrink=1, pad=0.02)
    cb2.ax.set(ylabel='RH / %')
    lat_2d = np.tile(
        eml_ds.lat, 
        len(eml_ds.fulllevel)).reshape(
            (len(eml_ds.fulllevel), len(eml_ds.lat)))
    c = axs.contour(
        lat_2d, eml_ds.pfull_mean/100, eml_ds.ua_mean,
        cmap='RdYlGn', levels=np.arange(-12, 14, 2),
        vmin=-12, vmax=14,
        linewidths=0.8,
        alpha=1, linewidth=2,
    )
    # cb2 = plt.colorbar(
    #     c, orientation="vertical", shrink=1, pad=0.02)
    # cb2.ax.set(ylabel='u-wind / ms$^{-1}$')
    axs.text(17, 550, 'AEJ', color='orangered')
    axs.invert_yaxis()
    axs.set_ylim([1013.25, 100.00])
    axs.set_ylabel('Pressure / hPa', fontsize=12)
    axs.set_xlabel('Latitude / deg', fontsize=12)
    axs.contour(
        lat_2d,
        eml_ds.pfull_mean/100, 
        eml_ds.ta_mean-273.15,
        levels=[0],
        linestyles='--', linewidths=1, colors='whitesmoke')
    axs.text(25, 550, '0° C', color='whitesmoke')
    axs.text(11, 950, 'ML', color='yellowgreen')
    ax2 = axs.twiny()
    eml_ds_15N = eml_ds.sel(lat=15)
    stability = static_stability(
        eml_ds_15N.pfull_mean*units.Pa, eml_ds_15N.ta_mean*units.kelvin)
    ax2.plot(stability*100, eml_ds_15N.pfull_mean/100, color='black')
    ax2.invert_yaxis()
    ax2.set(
        ylim=[1013.25, 100.00], xlim=[0, 1e-3], 
        xlabel='static stability at 15° N / 10$^{-3}$ K hPa$^{-1}$')
    ax2.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax2.set_xscale('log')
    plt.savefig(f'plots/paper/monsoon_african_circulation_cmap_2021-{month}.png', dpi=300)
if __name__ == '__main__':
    main()