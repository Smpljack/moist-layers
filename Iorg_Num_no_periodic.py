from netCDF4 import Dataset
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
from statsmodels.distributions.empirical_distribution import ECDF as ecdf


def Iorg(convective_array):
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
    return Iorg,num

def main():
    convective_array = np.random.randint(
        low=0, high=2, size=(10, 10))
    print(Iorg(convective_array))


if __name__ == '__main__':
    main()
