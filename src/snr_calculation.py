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
    Calculate the periodogram SNR of the best period 

    Assumes fixed frequency spacing for periods

    periodogram - the periodogram values

    periods     - periods associated with the above values

    index_to_evaluate - index of period to examine

    duration    - total duration of the observations
    
    per_type    - which period search algorithm was used

    (optional)
    freq_window_epsilon - sets the size of the exclusion area
               in the periodogram for the calculation
               
    rms_window_bin_size - number of points to include in
               calculating the RMS for the SNR
    """

    # Some value checking
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

    # Setting up some parameters
    freq_window_size = freq_window_epsilon/duration
    delta_frequency = abs(1./periods[1] - 1./periods[0])
    freq_window_index_size = int(round(freq_window_size/delta_frequency))

    # More value checking
    if freq_window_index_size > len(periodogram):
        raise ValueError("freq_window_index_size is greater than total periodogram length")
    elif freq_window_index_size > .9*len(periodogram):
        raise ValueError("freq_window_index_size is greater than 90% total length of periodogram")
    elif freq_window_index_size > .8*len(periodogram):
        print("here 80%")
        warnings.warn("freq_window_index_size is greater than 80% total length of periodogram")

    perdgm_window = [] # For storing values for RMS calculation

    # Which values to include in perdgm_window
    if index_to_evaluate > freq_window_index_size:
        perdgm_window.extend(periodogram[max(0,index_to_evaluate-freq_window_index_size-rms_window_bin_size+1):index_to_evaluate-freq_window_index_size+1].tolist())
    if index_to_evaluate + freq_window_index_size < len(periodogram):
        perdgm_window.extend(periodogram[index_to_evaluate+freq_window_index_size:index_to_evaluate+freq_window_index_size+rms_window_bin_size].tolist())
    perdgm_window = np.array(perdgm_window)

    # Include only finite values
    wherefinite = np.isfinite(perdgm_window)

    # Sigma clip
    vals, low, upp = sigmaclip(perdgm_window[wherefinite],low=3,high=3)

    # Calculate standard deviation
    stddev = np.std(vals)

    # Return
    if per_type.upper() == 'PDM': # If PDM, use correct amplitude
        return (1.-periodogram[index_to_evaluate])/stddev
    else:
        return periodogram[index_to_evaluate]/stddev
