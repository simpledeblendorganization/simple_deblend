'''snr_calculation.py - Joshua Wallace - Mar 2019

This code calculates the periodogram peak signal-to-noise ratio and
then compares to a (potentially) user-set threshold value to determine
robustness of the periodogram peak.
'''

from scipy.stats import sigmaclip
import numpy as np
from astrobase.lcmath import phase_magseries_with_errs
import copy
import bisect


def weighted_rms(vals,errs):
    """
    Thanks to https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """

    weights = 1./np.power(errs,2)

    average = np.average(vals, weights=weights)
    
    # Fast and numerically precise:
    variance = np.average((vals-average)**2, weights=weights)
    return np.sqrt(variance)



"""
def ____binnedrms(times,mags,errs,transit_duration):


    # Bin the magnitudes
    mintime = times.min()
    n_bins = np.ceil( (times.max()-mintime)/transit_duration)

    binned_mags = []
    for i in range(n_bins):
        binned_mags.append([])

    for t,m,e in zip(times,mags,errs):
        binned_mags[np.floor((t-mintime)/transit_duration)] = (m,e)

    # Calculate average magnitude in each bin
    avg_mags = []
    avg_mags_weightperbin = []
    for i in range(len(binned_mags)):
        if binned_mags[i]:
            this_bin_mags, this_bin_errs = zip(*binned_mags[i])
            avg_mags.append(np.average(this_bin_mags,weights=1/np.power(this_bin_errs,2)))
            avg_mags_weightperbin.append(np.sum(1./np.power(this_bin_errs,2)))

        

    avesum3 = np.sum(np.power(binsig,2))
    rmsbinthy = np.sqrt(avesum3 / n)

    return weighted_rms(np.array(avg_mags),np.array(avg_mags_weightperbin),
                        errsasweights=True)
"""


def dmax(a,b):
    if a >= b:
        return a
    else:
        return b

def dmin(a,b):
    if a <= b:
        return a
    else:
        return b


"""
def getrms(times,mags,errs):
  if len(times) > 0:
      avesum1 = 0.
      avesum2 = 0.
      avesum3 = 0.

      for i in range(len(times)):
          avesum1 += mag[i]
          avesum2 += mag[i]*mag[i]
          avesum3 += sig[i]*sig[i]

      if n > 0.:
	  ave = avesum1 / float(len(avesum1))
	  rmsthy = np.sqrt(avesum3 / float(len(avesum3)))
	  rmsval = np.sqrt((avesum2 / float(len(avesum2))) - (ave*ave))

      else:
	  rmsthy = -1.;
	  rmsval = -1.;

  else:
      rmsthy = -1.
      rmsval = -1.

  return(rmsval, rmsthy)
"""



def binnedrms(times,mags,errs,binsize):

    # Ensure sorted
    if any(np.diff(times)<0.):
        raise ValueError("Times not sorted")

    sumval1 = []
    sumval2 = []
    sumval3 = []

    for i in range(len(times)):
        if i == 0:
            sumval1.append(mags[i]/errs[i]**2)
            sumval2.append(1./errs[i]**2)
            sumval3.append(errs[i]**2)
        else:
            sumval1.append(sumval1[i-1] + mags[i]/errs[i]**2)
            sumval2.append(sumval2[i-1] + 1./errs[i]**2)
            sumval3.append(sumval3[i-1] + errs[i]**2)


    #jminold = 0
    #jmaxold = 0

    binmag = []
    binsig = []

    for i in range(len(mags)):
        bin_min = dmax(times[i] - binsize, times[0])
        bin_max = dmin(times[i] + binsize, times[-1])

        #Determine the id of the first point above the minimum t and the first point above the maximum t
        jmin = bisect.bisect_left(times,bin_min)
        jmax = bisect.bisect_right(times,bin_max)
        #jmax = jmax - 1

        jmax = (jmax-1 if times[jmax] > bin_max else jmax) if\
            jmax < len(times) else len(times)-1
        #jminold = jmin
        #jmaxold = jmax

        if jmin < len(times) and jmin > 0:
            v = sumval2[jmax] - sumval2[jmin-1]
            if v > 0:
                binmag.append((sumval1[jmax] - sumval1[jmin-1]) / v)
                binsig.append(np.sqrt((sumval3[jmax] - sumval3[jmin-1]) / float( (jmax - jmin + 1)**2)))
		
            else:
                binmag.append(0.)
                binsig.append(0.)

        elif jmin == 0 and sumval2[jmax] > 0.:
            binmag.append(sumval1[jmax] / sumval2[jmax])
            binsig.append(np.sqrt(sumval3[jmax] / float((jmax + 1)**2)))

        else:
            binmag.append(0.)
            binsig.append(0.)


    avesum1 = 0.
    avesum2 = 0.
    avesum3 = 0.
    n = 0
    for i in range(len(mags)):
        if binsig[i] > 0.:
            avesum1 += binmag[i]
            avesum2 += binmag[i] * binmag[i]
            avesum3 += binsig[i] * binsig[i]
            n += 1
    if n > 0:
        ave = avesum1 / float(n)
        rmsthy = np.sqrt(avesum3 / float(n))
        rmsval = np.sqrt((avesum2 / float(n)) - (ave * ave))
    else:
        rmsthy = -1.
        rmsval = -1.

    return (rmsval, rmsthy)



def redwhitenoise_calc(times,mags,errs,period,q):

    rmsval = weighted_rms(mags,errs)
    print("White noise: " + str(rmsval))

    rmsval_thy = np.sqrt(np.average(np.power(errs,2)))
    #rmsval, rmsval_thy =  getrms(times,mags,errs)

    rmsbinval, rmsbinval_thy = binnedrms(times,mags,errs,q*period)

    rmsbinval_expected = rmsval * rmsbinval_thy / rmsval_thy


    rednoise_squared = rmsbinval**2 - rmsbinval_expected**2
    if rednoise_squared > 0.:
        print("Red noise: " + str(np.sqrt(rednoise_squared)))
        return (rmsval, np.sqrt(rednoise_squared))
    else:
        print("Red noise:  0.")
        return (rmsval, 0.)



def pinknoise_calc(times,mags,errs,period,transitduration,depth,
                   npoints_transit,epoch,ntransits,magsarefluxes=False):

    # Get corrected BLS mags
    phased_magseries = phase_magseries_with_errs(times,
                                                 mags,
                                                 errs,
                                                 period,
                                                 epoch,
                                                 wrap=False,
                                                 sort=False)

    tphase = phased_magseries['phase']
    tmags = phased_magseries['mags']
    terrs = phased_magseries['errs']

    transitphase = transitduration/2.0
    transitindices = ((tphase < transitphase) |
                      (tphase > (1.0 - transitphase)))

    corrected_mags = copy.copy(tmags)
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

    

    whitenoise, rednoise = redwhitenoise_calc(times,corrected_mags,errs,period,transitduration)

    sigtopink2 = depth**2/(whitenoise**2/npoints_transit + rednoise**2/ntransits)

    return np.sqrt(sigtopink2)

 


if __name__ == "__main__":
    
    # Print a basic one
    t1 = np.linspace(0,100,5000)
    rand1 = np.random.RandomState(seed=1844)

    mags1 = 13. + 0.005*rand1.randn(5000)
    errs1 = np.array([0.005]*5000)

    print(pinknoise_calc(t1,mags1,errs1,3.4,.1,0.,int(.1*5000),
                         8.2,int(100/3.4)))
