from astropy.table import Table
import numpy as np
from astropy.convolution import convolve, Box1DKernel

def load_lc(fname):
    num_id = fname.split('-')[-1]
    filepath = f'/Users/z5214708/tess_data/stello/Kepler/LC/KBonus/kbonus-bkg/lcs/{num_id[0:4]}/{num_id}/hlsp_kbonus-bkg_kepler_kepler_{fname}_kepler_v1.0_lc.fits'


    lc = Table.read(filepath, hdu=1, format='fits', unit_parse_strict='silent')

    #remove NaNs
    nan_mask = np.isnan(lc['SAP_FLUX'])
    time = np.asarray(lc['TIME'][~nan_mask])
    flux = np.asarray(lc['SAP_FLUX'][~nan_mask])

    return time, flux


def split_light_curve(time, flux):
    '''
    Take a kepler light curve and split it into Kepler observing quarters.
    Parameters:
        time: an array like object
        flux: an array like object

    Returns:
        split_time, split_flux: a list of arrays
    '''
    step = [54953.5391589, 54964.5125410, 55002.5206148, 55093.2251257, 55185.3763099, 55276.4899601, 55372.4504619,
            55463.1754762, 55568.3628982, 55641.5155207, 55739.8464244, 55834.2088010, 55932.4081248, 56015.7365100,
            56107.1401772, 56206.4885332, 56305.0965127, 56392.2257117,70000]
    zpt = 54833
    step = [val - zpt for val in step]
    intervals = [(step[i], step[i + 1]) for i in range(len(step) - 1)]
    split_time = []
    split_flux = []
    for start, end in intervals:
        mask = (time > start) & (time <= end)
        if time[mask].size > 0:
            split_time.append(time[mask])
            split_flux.append(flux[mask])

    return split_time, split_flux

def segment_split(time,flux,ds_bad_times):
    # Split the light curve into stable segments
    stable_times = []
    stable_fluxes = []
    for i in range(len(ds_bad_times) - 1):
        start_time = ds_bad_times[i]
        end_time = ds_bad_times[i + 1]

        # Extract data within the current stable segment
        segment_mask = (time > start_time) & (time <= end_time)
        segment_time = time[segment_mask]
        segment_flux = flux[segment_mask]

        if len(segment_time) > 0:  # Skip empty segments
            stable_times.append(segment_time)
            stable_fluxes.append(segment_flux)

    return stable_times,stable_fluxes

def smooth(flux, width_days=2):
    #smooth by low pass filtering with a 2 day filter. this is the simplest version

    cadence = 29.4/(60 * 24) #cadence in days
    points_per_day = int(1 / cadence)

    filter_width = width_days * points_per_day  # 2 day high pass filter

    highly_smoothed_flux1 = convolve(flux, Box1DKernel(filter_width), boundary='extend')
    smoothed = convolve(highly_smoothed_flux1, Box1DKernel(filter_width), boundary='extend')

    return smoothed