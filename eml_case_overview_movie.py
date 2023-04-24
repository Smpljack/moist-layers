import intake
import matplotlib.pyplot as plt
import numpy as np
from typhon.physics import vmr2relative_humidity, specific_humidity2vmr, e_eq_mixed_mk
import seaborn as sns
from cartopy import crs as ccrs  # Cartogrsaphy library
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from decimal import Decimal
import argparse
import sys
sys.path.append('/home/u/u300676/moist-layers/')
import moist_layers as ml
import eval_eml_chars as eec
sns.set_context('paper')

parser = argparse.ArgumentParser()
parser.add_argument("--time_start", type=str,
                help="timestamp",
                default="2021-07-29T15:00:00")
parser.add_argument("--time_end", type=str,
                help="timestamp",
                default="2021-08-02T00:00:00")
args = parser.parse_args()
# Open the main DKRZ catalog
# Open the main DKRZ catalog
print(args.time_start, flush=True)
cat = intake.open_catalog(["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]


# Load a Monsoon 2.0 dataset and the corresponding grid
ds = cat["luk1043"].atm3d.to_dask()
grid = cat.grids[ds.uuidOfHGrid].to_dask()


ds_daily = (
    ds.sel(time=slice("2021-07-01", "2021-08-01"))
    .resample(time="1D", skipna=True)
    .mean()
)

# Define a EUREC4A mask
mask_eurec4a = (
    (grid.clon > np.deg2rad(-65)) &
    (grid.clon < np.deg2rad(-40)) &
    (grid.clat > np.deg2rad(5)) &
    (grid.clat < np.deg2rad(25))
)

mask_eml = (
    (grid.clon > np.deg2rad(-45.5)) &
    (grid.clon < np.deg2rad(-44.5)) &
    (grid.clat > np.deg2rad(8)) &
    (grid.clat < np.deg2rad(25))
)

mask_eml_point = (
    (grid.clon > np.deg2rad(-45.5)) &
    (grid.clon < np.deg2rad(-44.5)) &
    (grid.clat > np.deg2rad(21.5)) &
    (grid.clat < np.deg2rad(22.5))
)

ds_mean = ds_daily.isel(cell=mask_eurec4a).mean("cell").mean("time").load()

# TEST
# q = ds_eml.hus.isel(cell=mask_eml).isel(cell=1000).sel(time='2021-07-30T12:00:00')[42:][::-1].values
# vmr = specific_humidity2vmr(q)
# p = ds_eml.pfull.isel(cell=mask_eml).isel(cell=1000).sel(time='2021-07-30T12:00:00')[42:][::-1].values
# z = ds_eml.zg.isel(cell=mask_eml).isel(cell=1000)[42:][::-1].values

# ref_profile = ml.reference_h2o_vmr_profile(vmr, p, z, from_mixed_layer_top=True)

ds_eml_t = ds.sel(time=args.time_start, cell=mask_eml).load()
ds_eurec4a_t = ds.sel(time=args.time_start, cell=mask_eurec4a).load()
ds_point_t = ds.sel(time=args.time_start, cell=mask_eml_point).load() 
print('Data loaded.', flush=True)
sns.set_context('paper')
# fig, ax = plt.subplots(ncols=3, figsize=(10,5), sharey=False)
fig = plt.figure(figsize=(9,7))
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
data = ds_eurec4a_t.hus.isel(fulllevel=68).isel(cell=slice(0, None, 1)).values
lon = grid.clon.isel(cell=mask_eurec4a).isel(cell=slice(0, None, 1)).values
lat = grid.clat.isel(cell=mask_eurec4a).isel(cell=slice(0, None, 1)).values
cs1 = ax1.scatter(
        np.rad2deg(lon),
        np.rad2deg(lat),
        s=1,
        c=data,
        vmin=0,
        vmax=0.005,
        cmap="density",
        transform=ccrs.PlateCarree(),
    )
ax1.plot([-45, -45], [8, 25],
            color='red', linestyle='-',
            transform=ccrs.PlateCarree(),
            )
ax1.scatter(-45, 21, color=sns.color_palette('colorblind')[2], marker='x', s=100)
cb1 = plt.colorbar(cs1, extend="max", orientation="vertical", shrink=0.5, pad=0.01)
cb1.ax.set_ylabel("500 hPa Specific Humidity (kg/kg)", fontsize=7)
cb1.ax.tick_params(labelsize=10)
ax1.coastlines(resolution="10m", linewidth=0.6)
ax1.set_extent([-65, -40, 5, 25], crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(10, 26, 5), crs=ccrs.PlateCarree())
lat_formatter = LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.set_xticks(np.arange(-65, -39, 5), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.tick_params(labelsize=12)
ax1.set_xlabel(None)
ax1.set_ylabel(None)
ax1.set_aspect(0.5)
no_nan_p = ~np.any(ds_eml_t.pfull[42:, :].isnull().values, axis=0)
sort_lat = np.argsort(grid.isel(cell=mask_eml).isel(cell=no_nan_p).clat)

rh = vmr2relative_humidity(
    specific_humidity2vmr(ds_eml_t.hus[42:, sort_lat].values),
    ds_eml_t.pfull[42:, sort_lat].values, 
    ds_eml_t.ta[42:, sort_lat].values, 
e_eq=e_eq_mixed_mk) 
vmr_point = specific_humidity2vmr(ds_point_t.hus.mean('cell').values[42:])
ta_point = ds_point_t.ta.mean('cell').values[42:]
pfull_point = ds_point_t.pfull.mean('cell').values[42:]
zg_point = ds_point_t.zg.mean('cell').values[42:]
rh_point = vmr2relative_humidity(vmr_point, pfull_point, ta_point, e_eq=e_eq_mixed_mk)
ref_vmr_point = ml.reference_h2o_vmr_profile(
    vmr_point[::-1], pfull_point[::-1], zg_point[::-1], 
    from_mixed_layer_top=True)
ref_rh_point = vmr2relative_humidity(
    ref_vmr_point[::-1], pfull_point, ta_point, e_eq=e_eq_mixed_mk)
eml_chars = ml.eml_characteristics(
    vmr_point[::-1], ref_vmr_point, ta_point[::-1], pfull_point[::-1], zg_point[::-1], 
    lat=np.array([22]), lon=np.array([-45]), time=ds_eml_t.time.values, 
    min_eml_strength=0.2)
rh_mean = vmr2relative_humidity(
    specific_humidity2vmr(ds_mean.hus.values),
    ds_mean.pfull.values, 
    ds_mean.ta.values, 
e_eq=e_eq_mixed_mk)[42:]

rh_mean = np.array([rh_mean]*len(sort_lat)).T

ax4 = fig.add_subplot(222) 
ax4.plot(specific_humidity2vmr(ds_mean.hus.values)[42:], ds_mean.pfull[42:]/100, color='black', label='July 2021 mean')
ax4.plot(vmr_point, pfull_point/100, 
        label=r'$1°\times1°$ mean at X', color=sns.color_palette('colorblind')[2])
ax4.plot(ref_vmr_point[::-1], pfull_point/100, 
                label=r'ref profile at X', color=sns.color_palette('colorblind')[3])
ax4.invert_yaxis()
ax4.set_xscale('log')
ax4.set_yticklabels([])
ax4.set_xlim([0, 0.03])
ax4.set_ylim([1013.25, 100.00])
ax4.set_xlabel('H2O VMR', fontsize=12)
ax4.set_ylabel('Pressure / hPa', fontsize=12)
ax4.legend(loc='upper right')
ax2 = fig.add_subplot(223)
p = ax2.pcolormesh(np.rad2deg(np.array(
    [grid.isel(cell=mask_eml).isel(cell=no_nan_p).clat[sort_lat].values]*90))[42:, :], 
                ds_eml_t.pfull[42:, :].isel(cell=no_nan_p)[:, sort_lat].values/100, 
                (rh - rh_mean)*100, cmap='PuOr', vmin=-50, vmax=50)
ax2.plot([21, 21], [1013.25, 100],
            color=sns.color_palette('colorblind')[2], linestyle='-',
            )
cb2 = plt.colorbar(p, extend="max", orientation="vertical", shrink=1, pad=0.02)
# cb2.ax.set_ylabel(r"$RH-RH_{mean}$", fontsize=12)
cb2.ax.text(1, 50, r"$RH-RH_{mean}$")
cb2.ax.tick_params(labelsize=10)
ax2.invert_yaxis()
ax2.set_ylim([1013.25, 100.00])
ax2.set_ylabel('Pressure / hPa', fontsize=12)
ax2.set_xlabel('Latitude / deg', fontsize=12)
ax2.tick_params(labelsize=12)

ax3 = fig.add_subplot(224)
ax3.plot(rh_mean[:, 0]*100, ds_mean.pfull[42:]/100, color='black', label='July 2021 mean')
ax3.plot(rh_point*100, pfull_point/100, 
        label=r'$1°\times1°$ mean at X', color=sns.color_palette('colorblind')[2])
ax3.plot(ref_rh_point*100, pfull_point/100, 
                    label=r'ref profile at X', color=sns.color_palette('colorblind')[3])
ax3.invert_yaxis()
ax3.set_yticklabels([])
ax3.set_xlim([0, 100])
ax3.set_ylim([1013.25, 100.00])
ax3.set_xlabel('Relative Humidity / %', fontsize=12)
ax3.legend(loc='lower left')
if ~np.isnan(eml_chars.strength)[0]:
    eml_ds = eml_chars.to_xr_dataset()
    eml_ds_filtered = eec.filter_eml_data(
        eml_ds, 
        min_strength=0.2, 
        min_pmean=50000, 
        max_pmean=70000, 
        min_pwidth=10000, 
        max_pwidth=40000,
        )
    if len(eml_ds_filtered.eml_count) > 0:
        strongest = eml_ds_filtered.strength.values.argmax()
        ax3.text(50, 400, 'EML strength:'
                        f'{eml_ds_filtered.strength[strongest].values}', 
                fontsize=10)
        pmean_ind = abs(pfull_point - eml_ds_filtered.pmean[strongest].values).argmin()
        pmin_ind = abs(pfull_point - eml_ds_filtered.pmin[strongest].values).argmin()
        pmax_ind = abs(pfull_point - eml_ds_filtered.pmax[strongest].values).argmin()
        ax3.hlines(eml_ds_filtered.pmean[strongest]/100, 
                    rh_point[pmean_ind]*100-10, rh_point[pmean_ind]*100+10,
                    color=sns.color_palette('colorblind')[3])
        ax3.hlines(eml_ds_filtered.pmin[strongest]/100, 
                    rh_point[pmin_ind]*100-5, rh_point[pmin_ind]*100+5,
                    color=sns.color_palette('colorblind')[3])
        ax3.hlines(eml_ds_filtered.pmax[strongest]/100, 
                    rh_point[pmax_ind]*100-5, rh_point[pmax_ind]*100+5,
                    color=sns.color_palette('colorblind')[3]) 
    elif (len(eml_ds.eml_count) > 0):
        strongest = eml_ds.strength.values.argmax()
        if (eml_ds.strength[strongest] < 0.2):
            ax3.text(50, 400, 'EML not strong enough:\n'
                            f'{eml_ds.strength[strongest].values*100} %')
        elif eml_ds.pmean[strongest] < 50000:
            ax3.text(50, 400, 'EML pmean too low:\n'
                            f'{np.round(eml_ds.pmean[strongest].values/100, 2)} hPa')
        elif eml_ds.pmean[strongest] > 70000:
            ax3.text(50, 400, 'EML pmean too high:\n'
                            f'{np.round(eml_ds.pmean[strongest].values/100, 2)} hPa')
        elif eml_ds.pwidth[strongest] < 10000:
            ax3.text(50, 400, 'EML thickness too low:\n'
                            f'{np.round(eml_ds.pwidth[strongest].values/100, 2)} hPa')
        elif eml_ds.pwidth[strongest] > 30000:
            ax3.text(50, 400, 'EML thickness too high:\n'
                            f'{np.round(eml_ds.pwidth[strongest].values/100, 2)} hPa')
else: 
    ax3.text(70, 400, 'No EML') 
ax3.tick_params(labelsize=12)

fig.suptitle(f'ICON Monsoon Simulation, {args.time_start}')
plt.savefig(f'plots/eml_monsoon_overview_rh_eml_def_{args.time_start}.png', dpi=300)

eec.make_movie('plots/eml_monsoon_overview_rh_eml_def_*.png',
               'videos/eml_example_overview_movie_rh_eml_def.mp4')