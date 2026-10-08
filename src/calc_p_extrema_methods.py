import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from astropy.stats import sigma_clip
from clean_lc_paper_1 import *
from basics import *
import numpy as np

def calc_P_extrema_simple(time, flux):
    ###just how they do it in Yu 2020
    time = np.asarray(time)
    flux = np.asarray(flux)

    # Remove NaNs
    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]

    # Point-to-point differences
    dflux = np.diff(flux)

    # Iterative 4-sigma clipping
    clipped = sigma_clip(
        dflux,
        sigma=4.0,
        maxiters=None,
        masked=True
    )

    # Sign of each difference; retain masked positions
    signs = np.sign(clipped.data)
    good = ~clipped.mask & (signs != 0) #mask of good data points (i.e. not removed by sigma clipping) 

    # Count crossings only between adjacent unmasked differences
    adjacent_good = good[:-1] & good[1:]
    crossings = (
        adjacent_good &
        (signs[:-1] != signs[1:])
    )

    N_extrema = np.sum(crossings)

    if N_extrema == 0:
        return np.nan

    N = len(flux)
    dt = 29.4/(24 * 60)  # days

    return 2 * N * dt / N_extrema

def smooth(flux, width_days=2):
    #smooth by low pass filtering with a 2 day filter. this is the simplest version

    cadence = 29.4/(60 * 24) #cadence in days
    points_per_day = int(1 / cadence)

    filter_width = width_days * points_per_day  # 2 day high pass filter

    highly_smoothed_flux1 = convolve(flux, Box1DKernel(filter_width), boundary='extend')
    smoothed = convolve(highly_smoothed_flux1, Box1DKernel(filter_width), boundary='extend')

    return smoothed