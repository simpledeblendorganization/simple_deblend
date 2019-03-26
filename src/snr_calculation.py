'''snr_calculation.py - Joshua Wallace - Mar 2019

This code calculates the periodogram peak signal-to-noise ratio and
then compares to a (potentially) user-set threshold value to determine
robustness of the periodogram peak.
'''

from scipy.stats import sigmaclip
import numpy as np
import warnings


def periodogram_snr(periodogram,periods,index_to_evaluate,duration,per_type,
                    freq_window_epsilon=3.,rms_window_bin_size=100):
    """
    Assumes fixed frequency spacing for periods
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
    if per_type.upper() not in ['LS','PDM','BLS']:
        raise ValueError("Periodogram type " + per_type + " not recognized")

    #print(len(periodogram))
    # Now, calculate the SNR
    freq_window_size = freq_window_epsilon/duration
    delta_frequency = abs(1./periods[1] - 1./periods[0])
    freq_window_index_size = int(round(freq_window_size/delta_frequency))

    print(len(periodogram), freq_window_size, freq_window_index_size)

    if freq_window_index_size > len(periodogram):
        raise ValueError("freq_window_index_size is greater than total periodogram length")
    elif freq_window_index_size > .9*len(periodogram):
        raise ValueError("freq_window_index_size is greater than 90% total length of periodogram")
    elif freq_window_index_size > .8*len(periodogram):
        print("here 80%")
        warnings.warn("freq_window_index_size is greater than 80% total length of periodogram")

    perdgm_window = []
    if index_to_evaluate > freq_window_index_size:
        perdgm_window.extend(periodogram[max(0,index_to_evaluate-freq_window_index_size-rms_window_bin_size):index_to_evaluate-freq_window_index_size].tolist())
    if index_to_evaluate + freq_window_index_size < len(periodogram):
        perdgm_window.extend(periodogram[index_to_evaluate+freq_window_index_size:index_to_evaluate+freq_window_index_size+rms_window_bin_size].tolist())
    perdgm_window = np.array(perdgm_window)
    wherefinite = np.isfinite(perdgm_window)
    vals, low, upp = sigmaclip(perdgm_window[wherefinite],low=3,high=3)
    stddev = np.std(vals)

    # Return
    if per_type.upper() == 'PDM': # If PDM, use correct amplitude
        return (1.-periodogram[index_to_evaluate])/stddev
    else:
        return periodogram[index_to_evaluate]/stddev
