import numpy as np 
import xarray as xr
from skimage.measure import label, regionprops
import intake
from sklearn.neighbors import NearestNeighbors
from statsmodels.distributions.empirical_distribution import ECDF as ecdf
import argparse

import eval_eml_chars as eec
import moist_layers as ml


def iorg(convective_array):
    """
    Calculates the organisation index for a matrix where the convective regions
    = 1 and the non convective regions = 0.
    Finds centres of all convective regions and calculates nearest neighbour distribution
    Compares the distributution to the expected Poisson distribution for a 'random' state
    Value of 0.5 is random. 0.5 - 1 is organised. > 0.5 is more evenly distributed.
    """
    nx = np.shape(convective_array)[1]
    ny = np.shape(convective_array)[0]

    # Repeat domain above, below, left, right + corners to account for periodic boundaries
    big = np.zeros([ny,nx])
    big = convective_array
    #for i in range(3):
    #for j in range(3):
    #        big[:ny,j*nx:(j+1)*nx] = convective_array

    conn = label(big,4) # All individual updraghts accounting for connected grid points
    props = regionprops(conn) # calculate properties of all the identified storms
    # Find the centroids of each storm
    centroids = []
    for prop in props:
        centroids.append(prop['centroid'])

    # Find nearest neighbours for each storm in the large domain (accounting for the periodic BCs
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(centroids)
    distances, indices = nbrs.kneighbors(centroids)

    # Locate the centroids of storms within original domain
    centroids_in_domain = []
    nentities = len(centroids) # Number of storms
    for j in range(nentities):
        # If centroid of interest is in the original domain
        # if centroids[j][1] >= nx and centroids[j][1] < nx*2:
        #    centroids_in_domain.append(j)
        centroids_in_domain.append(j)

    # Nearest neighbour distribution for all storms in domain (having accounted for periodic BCs
    nnd = distances[centroids_in_domain][:,1]
    # Cumulative distribution function (CDF) evaluated at the points in xx
    cdf_nndx,xx = ecdf(nnd).y,ecdf(nnd).x

    # Theoretical CDF of nnd
    num = float(len(nnd)) # Number of storms in the original domain
    lam = num/(nx*ny) # entities in box should be equal to num
    dd = xx.copy()
    dd[0] = 0 # make first value 0 for cdf_theory to start from 0 (instead of -inf)
    #CDF for nearest-neighbor statistic in a Poisson point process is given by Weibull distribution of this form
    cdf_theoryx = 1-np.exp(-1*lam*np.pi*dd**2)

    # Take area under curve to find organisation index 
    Iorg = np.trapz(cdf_nndx,cdf_theoryx)
    return Iorg

def get_convective_array(rain_rates, rr_threshold):
    return np.where(rain_rates > rr_threshold, 1, 0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_start", type=str,
                    help="timestamp",
                    default="2021-07-28T00:00:00")
    parser.add_argument("--time_end", type=str,
                    help="timestamp",
                    default="2021-08-02T00:00:00")
    args = parser.parse_args()
    cat = intake.open_catalog(
        ["https://dkrz.de/s/intake"])["dkrz_monsoon_disk"]
    # Load a Monsoon 2.0 dataset and the corresponding grid
    ds2d = cat["luk1043"].atm2d.to_dask().sel(
        time=slice(args.time_start, args.time_end))
    grid = cat.grids[ds2d.uuidOfHGrid].to_dask()
    ds2d = ml.mask_eurec4a(ds2d, grid)
    grid = ml.mask_eurec4a(grid, grid)

    for time in ds2d.time.values:
        data2d = ds2d.sel(time=time)
        lat_grid = np.arange(
            data2d.clat.min(), data2d.clat.max(), np.deg2rad(0.1))
        lon_grid = np.arange(
            data2d.clon.min(), data2d.clon.max(), np.deg2rad(0.1))
        gridded_rr = eec.grid_monsoon_data(
            data2d, 'rain_gsp_rate', lat_grid, lon_grid)
        conv_array = get_convective_array(gridded_rr*3600, rr_threshold=10)
        io = iorg(conv_array)
        print(r"Iorg="
              f"{np.round(io, 2)}")
if __name__ == '__main__':
    main()