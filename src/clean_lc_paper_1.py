from astropy.table import Table
from astropy.io import fits
import numpy as np
from basics import *
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt

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

def cut_quarters(time_quarters, flux_quarters, cutoff):
    '''
    Cuts quarters where standard deviation is too high.

        Parameters:
            time_quarters: a list of arrays
            flux_quarters: a list of arrays
            cutoff: a float

        Returns:
            keep_times, keep_fluxes: a list of arrays (quarters to keep)
    '''
    stds = [np.std(flux) for flux in flux_quarters]

    # Discard segments where the standard deviation is NaN or > cutoff * minimum valid std
    keep_times = []
    keep_fluxes = []
    valid_stds = [std for std in stds if std > 0]  # Filter out NaN values for min calculation
    if not valid_stds:  # If all stds are NaN, return empty lists
        return keep_times, keep_fluxes

    min_std = min(valid_stds)

    for i in range(len(flux_quarters)):
        if len(flux_quarters[i]) > 1 and np.isfinite(stds[i]) and stds[i] < cutoff * min_std:
            keep_times.append(time_quarters[i])
            keep_fluxes.append(flux_quarters[i])

    return keep_times, keep_fluxes

def calc_rms(vals):
    return np.sqrt(np.mean((vals-np.mean(vals))**2))

def fill_single_cadence_gaps(time, flux, cadence=0.020434530917555094):
    '''
    Fills single cadence gaps in light curve.
        Parameters:
            time: an array like object
            flux: an array like object
            cadence: a float (default is Kepler Long Cadence)

        Returns:
            time, flux: arrays where single cadence gaps are filled
    '''

    # Define a tolerance for identifying single-cadence gaps
    tolerance = 0.0001  # Small value to handle floating-point precision

    # Calculate time differences
    time_diff = np.diff(time)

    # Identify gaps approximately equal to two cadences
    gap_indices = np.where((time_diff > 2 * cadence - tolerance) &
                           (time_diff < 2 * cadence + tolerance))[0]

    # Fill gaps by interpolating
    for idx in gap_indices:
        # Calculate missing time and interpolated flux
        missing_time = time[idx] + cadence
        interpolated_flux = (flux[idx] + flux[idx + 1]) / 2

        # Insert missing time and flux
        time = np.insert(time, idx + 1, missing_time)
        flux = np.insert(flux, idx + 1, interpolated_flux)

    return np.asarray(time), np.asarray(flux)

def detrend_clip(clip_time, clip_flux, mean_stable, rms_stable):
    '''
        Detrends flagged clip of light curve. Iteratively fits and removes polynomial of up to max_order 3,
        until rms of the clip is < 1.1 x rms of stable segment. Clip is removed if third order polynomial does
        not accomplish this.
        Parameters:
            clip_time: an array like object (to be detrended)
            clip_flux: an array like object (to be detrended)
            mean_stable (float): mean of stable segment
            rms_stable (float): rms of stable segment

        Returns: detrended clip_time and clip_flux
    '''
    order = 1
    flag = False
    max_order = 3

    while not flag and order <= max_order:
        # Fit polynomial of order {order}
        coefficients = np.polyfit(clip_time, clip_flux, order)
        polynomial = np.poly1d(coefficients)

        # Subtract polynomial
        polynomial_values = polynomial(clip_time)
        clip_flux = clip_flux - polynomial_values

        # Shift to match the rest of the light curve
        clip_flux = clip_flux + (mean_stable - np.mean(clip_flux))

        # Check RMS
        rms_clip = np.sqrt(np.mean((clip_flux - np.mean(clip_flux)) ** 2))
        rms_stable

        if rms_clip < 1.1 * rms_stable:
            flag = True
        else:
            order += 1

    if order > max_order:
        return [], [] # Segment cut if detrending fails

    return clip_time, clip_flux

def calc_ptp(flux):
    '''
    Calculates ptp array of flux
        Parameters:
            flux: an array like object

        Returns:
            ptp: an array like object (ptp array of flux)
    '''

    # Calculate ptp array
    ptp = np.zeros(len(flux))
    for j in range(1, len(flux) - 1):
        ptp[j] = flux[j] - np.mean([flux[j - 1], flux[j + 1]])

    ptp[0] = flux[0] - flux[1]
    ptp[-1] = flux[-1] - flux[-2]

    return ptp

def detrend(time, flux):
    segment_length = max(time) - min(time)

    # Masks for early and late parts
    early_mask = time < min(time) + 3
    late_mask = time > max(time) - 1.5
    first_half_mask = time < min(time) + segment_length / 2

    ptp = calc_ptp(flux)
    ptp_rms = calc_rms(ptp)

    # Initially assume whole segment is stable
    stable_mask = np.ones(len(time), dtype=bool)
    early_flag = False
    late_flag = False

    # Compute RMS ratios for early and late segments
    early_rms = calc_rms(flux[early_mask])
    early_rms_ratio = early_rms / ptp_rms if early_mask.any() else None

    late_rms = calc_rms(flux[late_mask])
    late_rms_ratio = late_rms / ptp_rms if late_mask.any() else None

    if segment_length < 6 and segment_length > 1:  # If segment is less than 1 day, just HP filter it
        # If segment is short, compare first half to second half to flag for detrending
        first_half_rms = calc_rms(flux[first_half_mask])
        if first_half_rms > 2 * ptp_rms:
            early_flag = True
            stable_mask = stable_mask & ~early_mask
    elif segment_length > 6:
        # else, compare first 3 days to rest
        if early_rms > 2 * ptp_rms:
            early_flag = True
            stable_mask = stable_mask & ~early_mask
        # and last 1.5 days to rest
        if late_rms > 2 * ptp_rms:
            late_flag = True
            stable_mask = stable_mask & ~late_mask

    # Apply high-pass filter to the stable region
    stable_time, stable_flux = apply_hp_filter(time.copy()[stable_mask], flux.copy()[stable_mask], 4)
    mean_stable = np.mean(stable_flux)
    rms_stable = calc_rms(stable_flux)
    late_time, late_flux = [], []
    early_time, early_flux = [], []

    if early_flag:
        # Detrend first 3 days if flagged, comparing to stable section to know when to stop
        early_time, early_flux = detrend_clip(
            time.copy()[early_mask], flux.copy()[early_mask], mean_stable, rms_stable
        )

    if late_flag:
        # Detrend last 1.5 days if flagged, comparing to stable section to know when to stop
        late_time, late_flux = detrend_clip(
            time.copy()[late_mask], flux.copy()[late_mask], stable_time, stable_flux
        )

    # Combine all segments
    time = np.concatenate([early_time, stable_time, late_time])
    flux = np.concatenate([early_flux, stable_flux, late_flux])

    return time, flux

def clean_lc(time,flux, ds_bad_times):
    '''
    Cleans KBonus lightcurves (where NaNs have already been removed)
        Parameters:
            time: an array like object
            flux: an array like object
            ds_bad_times: a list of 'bad' timestamps (determined by DS)
        Returns:
            time, flux: arrays where KBonus lightcurves are cleaned
    '''

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
            #Remove outliers
            ptp = calc_ptp(segment_flux)
            threshold = 3 * np.std(ptp)
            outliers_mask = np.abs(ptp) >= threshold
            segment_time = segment_time[~outliers_mask]
            segment_flux = segment_flux[~outliers_mask]

            #Normalize
            segment_flux = segment_flux / np.median(segment_flux)

            segment_time, segment_flux = detrend(segment_time, segment_flux)

            stable_times.append(segment_time)
            stable_fluxes.append(segment_flux)

    #Split LC into quarters and cut quarters where std is >1.3 x the minimum quarter std
    time, flux = np.concatenate(stable_times), np.concatenate(stable_fluxes)
    time_quarters, flux_quarters = split_light_curve(time, flux)
    time_quarters, flux_quarters = cut_quarters(time_quarters, flux_quarters,1.3)

    #Fill single cadence gaps
    time = np.concatenate(time_quarters)
    flux = np.concatenate(flux_quarters)
    time, flux = fill_single_cadence_gaps(time,flux)

    return time, flux



ds_bad_times = [131.43378, 181.60556, 230.40734, 255.43297, 256.19620, 258.60637, 280.65946, 281.18166, 290.66168,
                    321.59224, 383.57387, 397.03067, 400.72203, 442.41804, 475.19640, 503.95780, 538.38311, 566.70265,
                    598.47678, 629.44327, 660.45417, 690.46083, 711.14815, 761.27976, 802.52966, 845.14957, 874.27250,
                    886.76523, 887.44811, 906.04662, 937.05752, 969.43418, 1032.4560, 1063.3865, 1070.0949, 1099.2981,
                    1116.8522, 1118.6599, 1121.3914, 1122.0341, 1126.0511, 1154.0492, 1160.7576, 1182.1278, 1215.0626,
                    1245.0291, 1273.1879, 1305.0424, 1336.4148, 1371.4828, 1404.4980, 1435.5089, 1471.2598, 1525.0870,
                    1558.0662, 1559.1508, 1591.2463, 130.43898, 169.63898, 183.55898, 200.31608, 231.19898, 246.18889,
                    260.11898, 291.31898, 323.43898, 349.63898, 352.23898, 372.75898, 384.15898, 443.31898, 476.15898,
                    504.43898, 539.35898, 567.27898, 599.79898, 630.07898, 660.91898, 691.03898, 719.75898, 735.19898,
                    763.59898, 808.39898, 845.55898, 874.63898, 906.71898, 937.71898, 969.99898, 1001.0390, 1032.9590,
                    1064.0390, 1072.4390, 1098.4390, 1126.6390, 1154.7990, 1164.8790, 1182.5990, 1215.5990, 1245.7590,
                    1274.0790, 1289.7190, 1296.1990, 1306.0390, 1337.1190, 1373.2390, 1405.1990, 1413.0390, 1418.3190,
                    1436.1590, 1471.9590, 1477.5990, 1488.4390, 1525.9590, 1581.9190, 1586.6390]
ds_bad_times.sort()



def detrend_start_old(time, flux, check_days):
    '''
    Now defunct, but removes polynomials from first part of segment of light curve
        Parameters:
            time: an array like object
            flux: an array like object
            check_days: a float
        Returns:
            time, flux: arrays where polynomials are fitted and removed from first [check_days] part of segment of light curve
    '''

    # Isolate first five days of segment
    early_mask = np.asarray(time < (min(time) + check_days))

    # Calculate RMS of early segment and of the rest of the LC
    rms_early = np.sqrt(np.mean((flux[early_mask] - 1) ** 2))
    rms_late = np.sqrt(np.mean((flux[~early_mask] - 1) ** 2))

    # Check if first five days is slopey
    if rms_early > 1.1 * rms_late:
        order = 1
        flag = False
        max_order = 3
        while not flag and order <= max_order:
            flux_copy = flux.copy()

            # Fit polynomial of order {order} to early region
            coefficients = np.polyfit(time[early_mask], flux[early_mask], order)
            polynomial = np.poly1d(coefficients)

            # Subtract polynomial
            polynomial_values = polynomial(time[early_mask])
            flux_copy[early_mask] = flux_copy[early_mask] - polynomial_values

            # Shift to match rest of LC
            later_mask = np.asarray((time > (min(time) + check_days)) & (time < (min(time) + 1.3*check_days)))
            flux_copy[early_mask] = flux_copy[early_mask] + (np.mean(flux_copy[later_mask]) - np.mean(flux_copy[early_mask]))

            # Check RMS
            rms_early = np.sqrt(np.mean((flux_copy[early_mask] - 1) ** 2))
            rms_late = np.sqrt(np.mean((flux_copy[~early_mask] - 1) ** 2))

            if rms_early < 1.1 * rms_late:
                flag = True

            else:
                order += 1

        if order == 4:
            # If still not detrended, cut first segment
            flux = flux_copy[~early_mask]
            time = time[~early_mask]

        else:
            #Save detrended version
            flux = flux_copy

    return time, flux