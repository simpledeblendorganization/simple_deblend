'''snr_calculation.py - Joshua Wallace - Mar 2019

This code calculates the periodogram peak signal-to-noise ratio and
then compares to a (potentially) user-set threshold value to determine
robustness of the periodogram peak.
'''

from scipy.stats import sigmaclip
import numpy as np
from astrobase.lcmath import phase_magseries_with_errs
import copy


def weighted_rms(vals,errs,errsasweights=False):
    """
    Thanks to https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if errsasweights:
        weights = errs
    else:
        weights = 1./np.power(errs,2)

    average = np.average(vals, weights=weights)
    
    # Fast and numerically precise:
    variance = np.average((vals-average)**2, weights=weights)
    return np.sqrt(variance)




def binnedrms(times,mags,errs,period):


    # Bin the magnitudes
    mintime = times.min()
    n_bins = np.ceil( (times.max()-mintime)/period)

    binned_mags = []
    for i in range(n_bins):
        binned_mags.append([])

    for t,m,e in zip(times,mags,errs):
        binned_mags[np.floor((t-mintime)/period)] = (m,e)

    # Calculate average magnitude in each bin
    avg_mags = []
    avg_mags_weightperbin = []
    for i in range(len(binned_mags)):
        if binned_mags[i]:
            this_bin_mags, this_bin_errs = zip(*binned_mags[i])
            avg_mags.append(np.average(this_bin_mags,weights=1/np.power(this_bin_errs,2)))
            avg_mags_weightperbin.append(np.sum(1./np.power(this_bin_errs,2)))


    return weighted_rms(np.array(avg_mags),np.array(avg_mags_weightperbin),
                        errsasweights=True)

def redwhitenoise_calc(times,mags,errs,period):

    whitenoise = weighted_rms(corrected_mags,terrs)

    whitenoise_expected = np.sqrt(np.average(np.power(errs,2)))

    rmsbinval = binnedrms(times,mags,errs,period)

    rmsbinval_expected = 0. #whitenoise * rmsbinthy / whitenoise_expected


    return (whitenoise, np.sqrt(rmsbinval**2 - rmsbinval_expected**2))



def pinknoise_calc(times,mags,errs,period,transitduration,epoch,depth,magsarefluxes=False):

    # Get corrected BLS mags
    phased_magseries = phase_magseries_with_errs(times,
                                                 mags,
                                                 errs,
                                                 period,
                                                 epoch,
                                                 wrap=False,
                                                 sort=True)

    tphase = phased_magseries['phase']
    tmags = phased_magseries['mags']
    terrs = phased_magseries['errs']

    transitphase = transitduration/2.0
    transitindices = ((tphase < transitphase) |
                      (tphase > (1.0 - transitphase)))

    corrected_mags = copy(tmags)
    if magsarefluxes:
        # eebls.f returns +ve transit depth for fluxes
        # so add it in to get "flat" light curve
        corrected_mags[transitindices] = (
            corrected_mags[transitindices] + depth
            )
    else:
        # eebls.f returns -ve transit depth for magnitudes
        # so add it in to get "flat" light curve
        corrected_mags[transitindices] = (
            corrected_mags[transitindices] + depth
            )




    whitenoise, rednoise = redwhitenoise_calc()

    if rednoise < 0.:
        rednoise = 0.
