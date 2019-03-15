'''snr_calculation.py - Joshua Wallace - Mar 2019

This code calculates the periodogram peak signal-to-noise ratio and
then compares to a (potentially) user-set threshold value to determine
robustness of the periodogram peak.
'''

from scipy.stats import sigmaclip
import numpy as np

#num_periodogram_values_to_check_each_side = 15
median_filter_half_bin_size = 80
rms_window_half_bin_size = median_filter_half_bin_size
median_filter_half_bin_size_bls = 400
rms_window_half_bin_size_bls = median_filter_half_bin_size_bls
nbestpeaks = 12
periodepsilon = .05
freq_window_epsilon = 3.


def periodogram_snr(periodogram,periods,index_to_evaluate,duration,per_type):
    """
    """
    # Some checking
    if len(periodogram) != len(periods):
        raise ValueError("The lengths of the periodogram and the periods are not the same")
    if hasattr(index_to_evaluate,'__len__'):
        raise AttributeError("The index_to_evaluate has len attribute")
    if np.isnan(periodogram[index_to_evaluate]):
        raise ValueError("Selected periodogram value is nan")
    if np.isinf(periodogram[index_to_evaluate]):
        raise ValueError("Selected periodogram value is not finite")
    if pertype.upper() not in ['LS','PDM','BLS']:
        raise ValueError("Periodogram type " + pertype + " not recognized")

    # First, median-filter the periodogram
    corrected_perdgm = periodogram

    # Now, calculate the SNR
    freq_window_size = freq_window_epsilon/duration
    delta_frequency = 1./periods[0] - 1./periods[1]
    if delta_frequency < 0.:
        raise ValueError("The delta_frequency value is negative")
    freq_window_index_size = int(round(freq_window_size/delta_frequency))

    perdgm_window = []
    if per_type.upper() == 'BLS':
        h = rms_window_half_bin_size_bls
    else:
        h = rms_window_half_bin_size
    if index_to_evaluate > freq_window_index_size:
        perdgm_window.extend(corrected_perdgm[max(0,index_to_evaluate-freq_window_index_size-h):index_to_evaluate-freq_window_index_size].tolist())
    if index_to_evaluate + freq_window_index_size < len(corrected_perdgm):
        perdgm_window.extend(corrected_perdgm[index_to_evaluate+freq_window_index_size:index_to_evaluate+freq_window_index_size+h].tolist())
    perdgm_window = np.array(perdgm_window)
    wherefinite = np.isfinite(perdgm_window)
    vals, low, upp = sigmaclip(perdgm_window[wherefinite],low=3,high=3)
    stddev = np.std(vals)


    # Return
    if per_type.upper() == 'PDM': # If PDM, use correct amplitude
        return (1.-corrected_perdgm[index_to_evaluate])/stddev
    else:
        return corrected_perdgm[index_to_evaluate]/stddev
