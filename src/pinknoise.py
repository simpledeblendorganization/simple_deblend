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
        return (rmsval, np.sqrt(rednoise_squared))
    else:
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
            corrected_mags[transitindices] - depth
            )
    else:
        # eebls.f returns -ve transit depth for magnitudes
        # so add it in to get "flat" light curve
        corrected_mags[transitindices] = (
            corrected_mags[transitindices] - depth
            )

    

    whitenoise, rednoise = redwhitenoise_calc(times,corrected_mags,errs,period,transitduration)

    sigtopink2 = depth**2/(whitenoise**2/npoints_transit + rednoise**2/ntransits)

    return np.sqrt(sigtopink2)

 


if __name__ == "__main__":
    
    ################
    # Test a basic one
    print("First test, expecting WN 0.005, RN 0, SPN 0")
    t1 = np.linspace(0,100,5000)
    rand1 = np.random.RandomState(seed=1844)

    mags1 = 13. + 0.005*rand1.randn(5000)
    errs1 = np.array([0.005]*5000)

    spn = pinknoise_calc(t1,mags1,errs1,3.4,.1,0.,int(.1*5000),
                         8.2,int(100/3.4))
    print("SPN: " + str(spn))


    #################
    # Test a basic one w/ transit
    print("\nSecond test, expecting WN 0.007, RN 0, SPN 406")
    t2 = np.linspace(0,100,8000)
    rand2 = np.random.RandomState(seed=1847)

    q2 = 0.07
    depth2 = 0.12
    sigma2 = 0.007
    mags2 = 13. + sigma2*rand1.randn(8000)
    errs2 = np.array([sigma2]*8000)
    per2 = 5.7
    npt_t2 = 0
    epoch2 = 1.
    for i in range(len(t2)):
        phase = (t2[i] - epoch2)/per2 - np.floor((t2[i] - epoch2)/per2)
        if phase < q2/2. or phase > (1. - q2/2.):
            npt_t2 += 1
            mags2[i] += depth2
    spn = pinknoise_calc(t2,mags2,errs2,per2,q2,depth2,npt_t2,
                         epoch2,18)
    print("SPN: " + str(spn))


    ###################
    # Test a sawtooth with transit
    print("\nThird test, expecting WN 0.577, RN .3333, SPN 1.1997")
    t3 = np.linspace(0,100,8000)
    rand3 = np.random.RandomState(seed=1848)

    # straight line stddev = 0.5774080071035848
    # rmsval * rmsbinval_thy / rmsval_thy
    # 0.01 *  0.01/sqrt(npointsperbin, 25 or so)               /0.01
    #rednoise_squared = rmsbinval**2 - (0.01/5)**2
    # expect .333396

    q3 = 0.05
    depth3 = 0.1
    sigma3 = 0.01
    per3 = 2.*np.pi
    from scipy.signal import sawtooth
    saw3_amp = 1.
    mags3 = 10. + saw3_amp*sawtooth(t3) + sigma3*rand1.randn(8000)
    errs3 = np.array([sigma3]*8000)
    npt_t3 = 0
    epoch3 = 3.3
    for i in range(len(t3)):
        phase = (t3[i] - epoch3)/per3 - np.floor((t3[i] - epoch3)/per3)
        if phase < q3/2. or phase > (1. - q3/2.):
            npt_t3 += 1
            mags3[i] += depth3
    spn = pinknoise_calc(t3,mags3,errs3,per3,q3,depth3,npt_t3,
                         epoch3,16)
    print("SPN: " + str(spn))



    ###################
    # Test a sawtooth with transit, smaller sawtooth amplitude
    print("\nFourth test, expecting WN 0.288, RN ?, SPN ?")
    t4 = np.linspace(0,100,8000)
    rand4 = np.random.RandomState(seed=1848)

    # straight line stddev = 0.288704
    # rmsval * rmsbinval_thy / rmsval_thy
    # 0.01 *  0.01/sqrt(npointsperbin, 25 or so)               /0.01
    #rednoise_squared = rmsbinval**2 - (0.01/5)**2
    # expect .333396

    q4 = 0.05
    depth4 = 0.1
    sigma4 = 0.01
    per4 = 2.*np.pi
    from scipy.signal import sawtooth
    saw4_amp = 0.5
    mags4 = 10. + saw4_amp*sawtooth(t4) + sigma4*rand1.randn(8000)
    errs4 = np.array([sigma4]*8000)
    npt_t4 = 0
    epoch4 = 3.3
    for i in range(len(t4)):
        phase = (t4[i] - epoch4)/per4 - np.floor((t4[i] - epoch4)/per4)
        if phase < q4/2. or phase > (1. - q4/2.):
            npt_t4 += 1
            mags4[i] += depth4
    spn = pinknoise_calc(t4,mags4,errs4,per4,q4,depth4,npt_t4,
                         epoch4,16)
    print("SPN: " + str(spn))



    ###################
    # Test a sawtooth with transit, smaller sawtooth amplitude
    print("\nFifth test, expecting WN 0.0577, RN ?, SPN ?")
    t4 = np.linspace(0,100,8000)
    rand4 = np.random.RandomState(seed=1848)

    # straight line stddev = 0.0577408007103584
    # rmsval * rmsbinval_thy / rmsval_thy
    # 0.01 *  0.01/sqrt(npointsperbin, 25 or so)               /0.01
    #rednoise_squared = rmsbinval**2 - (0.01/5)**2
    # expect .333396

    q4 = 0.05
    depth4 = 0.1
    sigma4 = 0.01
    per4 = 2.*np.pi
    from scipy.signal import sawtooth
    saw4_amp = 0.1
    mags4 = 10. + saw4_amp*sawtooth(t4) + sigma4*rand1.randn(8000)
    errs4 = np.array([sigma4]*8000)
    npt_t4 = 0
    epoch4 = 3.3
    for i in range(len(t4)):
        phase = (t4[i] - epoch4)/per4 - np.floor((t4[i] - epoch4)/per4)
        if phase < q4/2. or phase > (1. - q4/2.):
            npt_t4 += 1
            mags4[i] += depth4
    spn = pinknoise_calc(t4,mags4,errs4,per4,q4,depth4,npt_t4,
                         epoch4,16)
    print("SPN: " + str(spn))
