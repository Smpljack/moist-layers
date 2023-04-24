import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import intake
import argparse
import verde as vd 
from skimage.measure import label, regionprops

import moist_layer_correlations as mlc
import moist_layers as ml
import eval_eml_chars as eec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_start", type=str,
                    help="timestamp",
                    default="2021-07-29T00:00:00")
    parser.add_argument("--time_end", type=str,
                    help="timestamp",
                    default="2021-08-11T00:00:00")
    args = parser.parse_args()
    # Open the main DKRZ catalog
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]

    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds3d = cat["luk1043"].atm3d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    ds2d = cat["luk1043"].atm2d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    grid = cat.grids[ds3d.uuidOfHGrid].to_dask()
    ds3d = ml.mask_warmpool(ds3d, grid)
    ds2d = ml.mask_warmpool(ds2d, grid)
    grid = ml.mask_warmpool(grid, grid)
    times = ds3d.time.values
    total_eml_areas = []
    mean_eml_strength = []
    mean_eml_height = []
    mean_eml_thickness = []
    iorgs = []
    for i, time in enumerate(times):
        print(str(time)[:19], flush=True)
        eml_ds = eec.load_eml_data(time)
        eml_ds = eec.filter_eml_data(
            eml_ds, 
            min_strength=1e-3, 
            min_pmean=50000, 
            max_pmean=70000, 
            min_pwidth=10000, 
            max_pwidth=30000,
            )
        region1 = (160, 180, 0, 20)
        region2 = (-180, -140, 0, 20)
        spacing = 0.1
        grid1 = vd.grid_coordinates(region1, spacing=spacing)
        grid2 = vd.grid_coordinates(region2, spacing=spacing)
        grid = (np.concatenate([grid1[0], grid2[0]], axis=1), 
                np.concatenate([grid1[1], grid2[1]], axis=1))
        heights = np.arange(2000, 7000, 100)
        eml_labels3d = eec.get_3d_height_eml_labels(eml_ds, grid, spacing, 
                                                    heights=heights)
        eml_labels2d = eec.project_3d_labels_to_2d(eml_labels3d)
        # eml_props = regionprops(eml_labels2d)
        # big_emls = [props.label for props in eml_props 
        #             if props.area > 500]
        # eml_labels2d = np.where(
        #     np.isin(eml_labels2d, big_emls), eml_labels2d, 0)
        eml_props = regionprops(eml_labels2d)
        total_eml_areas.append(
            np.sum([props.area for props in eml_props]))
        mean_eml_strength.append(eml_ds.strength.mean())
        mean_eml_height.append(eml_ds.pmean.mean())
        mean_eml_thickness.append(eml_ds.pwidth.mean())
        data2d = ds2d.sel(time=time)
        lat_grid = np.arange(
            data2d.clat.min(), data2d.clat.max(), np.deg2rad(0.1))
        lon_grid = np.arange(
            data2d.clon.min(), data2d.clon.max(), np.deg2rad(0.1))
        gridded_rr = eec.grid_monsoon_data(
            data2d, 'rain_gsp_rate', lat_grid, lon_grid)
        conv_array = mlc.get_convective_array(gridded_rr*3600, rr_threshold=10)
        iorgs.append(mlc.iorg(conv_array))


    total_eml_areas = np.array(total_eml_areas)
    domain_area = regionprops(np.ones(grid[0].shape, dtype=np.int64))[0].area
    fig, axs = plt.subplots(nrows=6, figsize=(10, 15), sharex=True)
    axs[0].plot(ds2d.time, ds2d.rain_gsp_rate.mean('cell')*3600)
    axs[0].set(ylabel='rain rate /\nmm hour$^{-1}$')
    axs[1].plot(ds3d.time, iorgs)
    axs[1].set(ylabel='I$_{org}$ /\n-')
    axs[2].plot(ds3d.time, total_eml_areas/domain_area)
    axs[2].set(ylabel='EML area fraction /\n-')
    axs[3].plot(ds3d.time, mean_eml_strength)
    axs[3].set(ylabel='mean EML strength /\n-')
    axs[4].plot(ds3d.time, mean_eml_height)
    axs[4].set(ylabel='mean EML height /\nkm')
    axs[5].plot(ds3d.time, mean_eml_thickness)
    axs[5].set(ylabel='mean EML thickness /\nkm')
    
    plt.savefig('plots/warmpool_domain_average_timeseries.png', dpi=300)

if __name__ == '__main__':
    main()
