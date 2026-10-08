import re
from astropy.table import Table
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from astropy.timeseries import LombScargle


def number_to_kic_id(num):
    '''
    Takes an integer (representing the kic/gaia id) and converts it to a string in the correct format

    eg: num: 757149 --> output: "kic-000757149"
        num: 2126013003413000448 --> output: "gaia-dr3-2126013003413000448"
    '''
    if num > 10 ** 10:
        output_string = f'gaia-dr3-{int(num)}'
    else:
        output_string = f'kic-{str(int(num)).zfill(9)}'

    return output_string

def process_flux(data, flux_column,hp=True):
    data = data[np.isfinite(np.asarray(data[flux_column]))] # remove Nans
    clip = sigma_clip(data[flux_column], sigma=4, maxiters=1)
    time = np.asarray(data['TIME'][~clip.mask])
    flux = np.asarray(data[flux_column][~clip.mask])
    if len(flux) > 0:
        flux = flux / np.median(flux)
        if hp:
            time, flux = apply_hp_filter(time, flux, 4)
    return time,flux

def apply_hp_filter(time, flux, width_days, sigma_flag=True, sigma_width=3):
    try:
        cadence = np.median(np.diff(time))  #Time between points (in days)
        points_per_day = int(1 / cadence)
    except ValueError:
        print('Something went wrong with the high pass filter')
        return [],[]
    filter_width = width_days * points_per_day  # 10 day high pass filter

    highly_smoothed_flux1 = convolve(flux, Box1DKernel(filter_width), boundary='extend')
    highly_smoothed_flux2 = convolve(highly_smoothed_flux1, Box1DKernel(filter_width), boundary='extend')
    hp_filtered_flux = flux / highly_smoothed_flux2

    if sigma_flag:
        clipped_flux = sigma_clip(hp_filtered_flux, sigma=sigma_width, maxiters=1)

        time = time[~clipped_flux.mask]
        hp_filtered_flux = hp_filtered_flux[~clipped_flux.mask]

    return time, hp_filtered_flux

def calc_lomb_scargle(t, y, fmax=1, use_nyquist=True):
    oversample = 1
    df = 1.0 / (max(t) - min(t))
    fmin = df

    if use_nyquist == True:
        ## calculate this as the nyquist frequency
        # binned to 5 days, so the nyquist in c/d is 1/(2*5) = 0.1
        res_days = np.median(np.diff(t))
        nyquist_cd = 1 / (2 * res_days)
        fmax = nyquist_cd

    else:
        fmax = fmax  # maximum frequency in cycles per day (c/d)

    freq = np.arange(fmin, fmax, df / oversample)
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    return freq * 11.574, sc * (1E12)  # return frequency in muHz and power in ppm^2


#DS spectra
def search_for_ds_spectrum(kic_id):
    full_file_path = '-'
    paths_to_ds = '/Users/z5214708/Desktop/Catalogs/list_kplr_all_llc.ts.clip4dpassfill.fft.pds.fullpath'
    with open(paths_to_ds, 'r') as f:
        file_paths = f.readlines()
    for file_path in file_paths:
        if kic_id in file_path:
            full_file_path = '/Users/z5214708/tess' + file_path.strip()
    return full_file_path

#Raw fits
def search_for_fits(kic_id):
    full_file_path = '-'
    paths_to_fits = '/Users/z5214708/Desktop/store_temp/Catalogs/paths_to_fits.txt'
    with open(paths_to_fits, 'r') as f:
        file_paths = f.readlines()
    for file_path in file_paths:
        if kic_id in file_path:
            full_file_path = file_path.strip()

    return full_file_path

def search_for_lc_DS(kic_id):
    full_file_path = '-'
    paths_to_ds = '/Users/z5214708/data/stello/Kepler/LC/KBonus/kbonus-bkg/list_kplr_all_llc.ts.clip4dpassfill.fullpath'
    with open(paths_to_ds, 'r') as f:
        file_paths = f.readlines()
    for file_path in file_paths:
        if kic_id in file_path:
            full_file_path = '/Users/z5214708' + file_path.strip()
    return full_file_path

def search_for_ps_DS(kic_id):
    paths_to_ds = '/Users/z5214708/data/stello/Kepler/LC/KBonus/kbonus-bkg/list_kplr_all_llc.ts.clip4dpassfill.fft.pds.fullpath'
    with open(paths_to_ds,'r') as f:
        file_paths = f.readlines()
    for file_path in file_paths:
        if kic_id in file_path:
            full_file_path = '/Users/z5214708' + file_path.strip()
            full_file_path = full_file_path.replace('lcs_4d', 'pds_4d')
    return full_file_path

def extract_id(file_path):
    match = re.search(r'(kic-\d{9}|gaia-dr3-\d+)',file_path)
    return match.group(0) if match else None

def filepath_to_four_data(filepath):
    stitched = Table.read(filepath, format='fits', unit_parse_strict='silent', hdu=1)
    psf_stitched = {}
    sap_stitched = {}
    psf_stitched['time'], psf_stitched['flux'] = process_flux(stitched, 'FLUX')
    sap_stitched['time'], sap_stitched['flux'] = process_flux(stitched, 'SAP_FLUX')

    # Quarters
    hdul = fits.open(filepath)
    quarters = [Table.read(filepath, format='fits', unit_parse_strict='silent', hdu=i) for i in range(len(hdul))
                if 'LIGHTCURVE_Q' in hdul[i].name]

    psf_q = {}
    sap_q = {}
    psf_q['time'] = np.concatenate([process_flux(q, 'FLUX')[0] for q in quarters])
    psf_q['flux'] = np.concatenate([process_flux(q, 'FLUX')[1] for q in quarters])
    sap_q['time'] = np.concatenate([process_flux(q, 'SAP_FLUX')[0] for q in quarters])
    sap_q['flux'] = np.concatenate([process_flux(q, 'SAP_FLUX')[1] for q in quarters])

    try:
        # Calculate power spectra
        psf_stitched['freq'], psf_stitched['power'] = calc_lomb_scargle(psf_stitched['time'], psf_stitched['flux'])
        sap_stitched['freq'], sap_stitched['power'] = calc_lomb_scargle(sap_stitched['time'], sap_stitched['flux'])
        psf_q['freq'], psf_q['power'] = calc_lomb_scargle(psf_q['time'], psf_q['flux'])
        sap_q['freq'], sap_q['power'] = calc_lomb_scargle(sap_q['time'], sap_q['flux'])
    except ValueError as e:
        print(f"Can't lomb scargle: {e}")
        return [], [], [], []

    return psf_stitched, sap_stitched, psf_q, sap_q


def quality_flags(quality):
    KEPLER_QUALITY_FLAGS = {
        "1": "Attitude tweak",
        "2": "Safe mode",
        "4": "Coarse point",
        "8": "Earth point",
        "16": "Zero crossing",
        "32": "Desaturation event",
        "64": "Argabrightening",
        "128": "Cosmic ray",
        "256": "Manual exclude",
        "1024": "Sudden sensitivity dropout",
        "2048": "Impulsive outlier",
        "4096": "Argabrightening",
        "8192": "Cosmic ray",
        "16384": "Detector anomaly",
        "32768": "No fine point",
        "65536": "No data",
        "131072": "Rolling band",
        "262144": "Rolling band",
        "524288": "Possible thruster firing",
        "1048576": "Thruster firing"
    }
    flag = 256
    if quality & flag >0:
        return ''

    flags = []
    for flag in KEPLER_QUALITY_FLAGS.keys():
        if quality & int(flag) > 0:
            flags.append(KEPLER_QUALITY_FLAGS[flag])
    return flags

def adv_process_flux(data, flux_column, hp=True):
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

    # Remove NaNs
    data = data[np.isfinite(np.asarray(data[flux_column]))]

    time = data['TIME']
    flux = data[flux_column]

    if len(flux) > 0:
        flux = flux / np.median(flux)  # Normalize flux

        ptp = np.zeros(len(flux))
        for i in range(1, len(flux) - 1):
            ptp[i] = flux[i] - np.mean([flux[i - 1], flux[i + 1]])

        ptp[0] = flux[0] - flux[1]
        ptp[-1] = flux[-1] - flux[-2]

        rms_ptp_total = np.sqrt(np.mean(ptp ** 2))

        # Filter based on ds_bad_times criteria
        mask = np.ones(len(time), dtype=bool)
        for bad_time in ds_bad_times:
            # Identify indices within 1.5d before and 2.5d after each bad timestamp
            after_indices = (time >= bad_time) & (time < bad_time + 2.5)
            before_indices = (time > bad_time - 1.5) & (time <= bad_time)

            # Calculate RMS for the regions before and after
            if np.any(after_indices):
                after_rms = np.sqrt(np.mean(ptp[after_indices] ** 2))
                if after_rms > 2 * rms_ptp_total:
                    mask[after_indices] = False

            if np.any(before_indices):
                before_rms = np.sqrt(np.mean(ptp[before_indices] ** 2))
                if before_rms > 2 * rms_ptp_total:
                    mask[before_indices] = False

        # Apply mask to time and flux
        time = np.asarray(time[mask])
        flux = np.asarray(flux[mask])

        # Apply high-pass filter if hp is True
        if hp:
            time, flux = apply_hp_filter(time, flux, 4)

        # Fill single-cadence gaps
        time_diff = np.diff(time)
        median_diff = np.median(time_diff)

        # While loop to handle dynamic insertion
        single_gap_indices = np.where(time_diff > median_diff * 1.5)[0]
        i = 0  # Initialize index for the while loop

        while i < len(single_gap_indices):
            idx = single_gap_indices[i]

            # Check if it's a single cadence gap with valid neighbors
            if idx + 1 < len(time) - 1 and time_diff[idx + 1] <= median_diff * 1.5:
                # Interpolate and insert
                missing_time = time[idx] + median_diff
                interpolated_flux = (flux[idx] + flux[idx + 1]) / 2

                # Insert into time and flux arrays
                time = np.insert(time, idx + 1, missing_time)
                flux = np.insert(flux, idx + 1, interpolated_flux)

                # Recalculate time_diff and single_gap_indices to update after insertion
                time_diff = np.diff(time)
                single_gap_indices = np.where(time_diff > median_diff * 1.5)[0]

                # Reset to 0 to check all new gaps from the start
                i = 0
            else:
                # Move to next index if no insertion was done
                i += 1


    return np.asarray(time), np.asarray(flux)