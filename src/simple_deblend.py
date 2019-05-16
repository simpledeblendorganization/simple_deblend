'''simple_deblend.py - John Hoffman & Joshua Wallace - Feb 2019

This code is where the period search results are calculate, analyzed,
and taken care of
'''


from scipy.stats import sigmaclip
from scipy.special import gammaln
import numpy as np
import math
import snr_calculation as snr
import pinknoise
import copy

from multiprocessing import Pool



def snr_threshold_tocomp(f,period=None):
    '''
    Wrapper function to agnostically return either a
    straight value or a value from a function

    f - either the value or the function

    period - (necessary if f is a function) the period
             to give to the function f; ignored if f is a value
    '''
    if callable(f):
        return f(period)
    else:
        return f


def _2d(arr):
    ''' Reshape 1d array to 2d array '''
    return np.reshape(arr, (len(arr), 1))


def design_matrix(t, freq, nharms):
    '''
    Design matrix for truncated Fourier series
    Parameters
    ----------
    t: array_like
        Time coordinates
    freq: float
        Frequency of signal
    nharms: int
        Number of harmonics to use to model unmodulated signal
    Returns
    -------
    X: np.ndarray
        Design matrix, shape = ``(N, 2 * nharms + 1)``, where
        ``N`` is the number of observations, and ``nharms`` is the
        number of harmonics
    '''

    omega = 2 * np.pi * freq
    phi = omega * t

    m_0 = np.ones_like(t)

    # mean magnitude
    cols = [m_0]

    # (unmodulated) signal
    for n in range(1, nharms + 1):
        for func in [np.cos, np.sin]:
            cols.append(func(n * phi))

    cols = list(map(_2d, cols))

    return np.hstack(cols)

class FourierFit(object):
    '''
    Class for fitting a Fourier series to a light curve

    nharmonics - the number of harmonics to use in the fit
    
    '''
    def __init__(self, nharmonics=5):
        self.nharmonics = nharmonics

    def fit(self, t, y, dy, freq, nharmonics=None,
            dy_as_weights=False,
            y_as_flux=False):
        '''
        Fit a Fourier series

        t - times of the data

        y - values of the data

        dy - errors on the data

        freq - frequency to fit on

        nharmonics - (optional) number of harmonics to fit on

        dy_as_weights - (optional) whether to use the dy values
                        as weights for the fit

        y_as_flux - (optional) whether the y values are flux
                    values instead of magnitudes
        '''

        # Some initialization stuff
        if nharmonics is not None:
            self.nharmonics = nharmonics

        self.y_as_flux = y_as_flux
        self.freq = freq

        X = design_matrix(t, self.freq, self.nharmonics)

        # cov^(-1)
        W = None
        if dy_as_weights:
            W = np.diag(dy)
        else:
            W = np.diag(np.power(dy, -2))

        # WLS: (X.T @ W @ X + V) @ theta = (X.T @ W) @ y
        # where V = \Sigma^(-1) is the *prior* covariance matrix
        # of the model parameters
        z = (X.T @ W) @ y

        S = X.T @ W @ X# + self.V

        # Parameters of the fit
        self.params = np.linalg.solve(S, z)

        return self

    #@property
    def flux_amplitude(self,return_df_f0=False):
        # delta flux

        phase = np.linspace(0, 1, 200)

        yfit = self(phase, in_phase=True) + self.params[0]

        if self.y_as_flux:
            return 0.5 * (max(yfit) - min(yfit))

        mbrite = min(yfit)
        mdim = max(yfit)

        delta_f = 10 ** (-0.4 * mbrite) - 10 ** (-0.4 * mdim)
        if return_df_f0:
            df_f0 = 10**(0.4 * (mbrite+mdim)/2.) * delta_f
            print("----  " + str(mbrite) + "  " + str(mdim) + "  "  + str(delta_f) + "    " + str(df_f0))
            return (delta_f, df_f0)
        else:
            return delta_f


    @classmethod
    def from_params(cls, c, s, offset, freq,
                    y_as_flux=False):

        ff = cls(nharmonics=len(c))
        ff.y_as_flux = y_as_flux
        ff.freq = freq
        ff.params = [offset]
        for C, S in zip(c, s):
            ff.params.extend([C, S])

        return ff

    def __call__(self, t, in_phase=False):
        # Return the value of the fit Fourier series at t
        freq = 1 if in_phase else self.freq
        X = design_matrix(t, freq, self.nharmonics)
        return X @ self.params - self.params[0]

def fap_baluev(t, dy, z, fmax, d_K=3, d_H=1, use_gamma=True, use_log=True):
    """
    False alarm probability for periodogram peak
    based on Baluev (2008) [2008MNRAS.385.1279B]

    Parameters
    ----------
    t: array_like
        Observation times.
    dy: array_like
        Observation uncertainties.
    z: array_like or float
        Periodogram value(s)
    fmax: float
        Maximum frequency searched
    d_K: int, optional (default: 3)
        Number of degrees of fredom for periodgram model.
        2H - 1 where H is the number of harmonics
    d_H: int, optional (default: 1)
        Number of degrees of freedom for default model.
    use_gamma: bool, optional (default: True)
        Use gamma function for computation of numerical
        coefficient; replaced with scipy.special.gammaln
        and should be stable now
    use_log: bool, optional (default: True)
        Return (natural) logarithm of false alarm probability

    Returns
    -------
    fap: float
        False alarm probability

    Example
    -------
    >>> rand = np.random.RandomState(100)
    >>> t = np.sort(rand.rand(100))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * 10. * t)
    >>> dy = 0.01 * np.ones_like(y)
    >>> y += dy * rand.rand(len(t))
    >>> proc = LombScargleAsyncProcess()
    >>> results = proc.run([(t, y, dy)])
    >>> freqs, powers = results[0]
    >>> fap_baluev(t, dy, powers, max(freqs))
    """

    N = len(t)
    d = d_K - d_H

    N_K = N - d_K
    N_H = N - d_H
    g = (0.5 * N_H) ** (0.5 * (d - 1))

    if use_gamma:
        g = np.exp(gammaln(0.5 * N_H) - gammaln(0.5 * (N_K + 1)))

    w = np.power(dy, -2)

    tbar = np.dot(w, t) / sum(w)
    Dt = np.dot(w, np.power(t - tbar, 2)) / sum(w)

    Teff = np.sqrt(4 * np.pi * Dt)

    W = fmax * Teff
    A = (2 * np.pi ** 1.5) * W

    if (0.5 * N_K * np.log10(z) < -2):
        log_gamma = gammaln(0.5 * N_H) - gammaln(0.5 * (N_K + 1))
        log_A_fmax = np.log(2) + 1.5 * np.log(np.pi) + np.log(W)

        log_tau = log_gamma - np.log(2 * np.pi) + log_A_fmax

        log_tau += 0.5 * (d - 1) * np.log(z / np.pi)

        log_tau += 0.5 * (N_K - 1) * np.log(1 - z)

        return (log_tau if use_log else np.exp(log_tau))

    eZ1 = (z / np.pi) ** 0.5 * (d - 1)
    eZ2 = (1 - z) ** (0.5 * (N_K - 1))

    tau = (g * A / (2 * np.pi)) * eZ1 * eZ2

    Psing = 1 - (1 - z) ** (0.5 * N_K)

    fap = 1 - Psing * np.exp(-tau)

    return np.log(fap) if use_log else fap

def _median_filtering_one_job(task):
    '''
    Median filter worker function for parallelization,
    works on determining the median filter

    task - the task being passed to this worker, see
           median_filtering function for details
    '''

    # Extract parameters
    (i,periodogramvals,freq_window_index_size,median_filter_size) = task

    window_vals = [] # For storing values to use in median filtering

    # Get the values to be used for the filter
    if i >= freq_window_index_size:
        if i - freq_window_index_size < 0:
            raise RuntimeError("Too small, " + str(i-freq_window_index_size))
        window_vals.extend(periodogramvals[max(0,i-freq_window_index_size-median_filter_size+1):i-freq_window_index_size+1].tolist())
    if i + freq_window_index_size < len(periodogramvals):
        window_vals.extend(periodogramvals[i+freq_window_index_size:i+freq_window_index_size+median_filter_size].tolist())
    window_vals = np.array(window_vals)

    # Keep only finite ones
    wherefinite = np.isfinite(window_vals)

    # Sigma clipping
    vals, low, upp = sigmaclip(window_vals[wherefinite],low=3,high=3)

    # Return the median value
    return np.median(vals)


def median_filtering(periodogramvals,periods,freq_window_epsilon,median_filter_size,
                     duration,which_method,nworkers=1):
    '''Median filter a periodogram

    periodogramvals - the periodogram values

    periods - the periods associated with periodogramvals

    freq_window_epsilon - sets the size of the exclusion area
               in the periodogram for the calculation

    median_filter_size - number of points to include in
               calculating the RMS for the SNR

    duration - total duration of the observations

    which_method - which period search algorithm was used

    nworkers - (optional) number of worker processes
    '''


    # First, make sure all the frequency values are equally spaced
    diff = np.diff(1./periods)
    for val in diff:
        if not math.isclose(val,diff[0],rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError("The frequency differences are not equal spacing, which this function assumes")

    # Initializing some values
    freq_window_size = freq_window_epsilon/duration
    freq_window_index_size = int(round(freq_window_size/abs(1./periods[0] - 1./periods[1])))

    # Set up processing pool
    pool = Pool(nworkers)

    # Set up tasks for processing
    tasks = [(i,periodogramvals,freq_window_index_size,median_filter_size) for i in range(len(periodogramvals))]

    # Run, collect results
    median_filter_values = pool.map(_median_filtering_one_job,tasks)

    pool.close()
    pool.join()
    del pool

    # Return median-filtered periodogram
    if which_method == 'PDM':
        return periodogramvals + (1. - np.array(median_filter_values))
    else:
        return periodogramvals - np.array(median_filter_values)


def iterative_deblend(t, y, dy, neighbors,
                      period_finding_func,
                      results_storage_container,
                      which_method,
                      function_params=None,
                      nharmonics_fit=5,
                      nharmonics_resid=10,
                      ID=None,
                      medianfilter=False,
                      freq_window_epsilon_mf=1.,
                      freq_window_epsilon_snr=1.,
                      window_size_mf=40,
                      window_size_snr=40,
                      snr_threshold=0.,
                      spn_threshold=None,
                      fap_baluev_threshold=None,
                      neighbor_peaks_tocheck=8,
                      max_blend_recursion=8,
                      recursion_level=0,
                      nworkers=1,
                      max_neighbor_amp=3.):
    """
    Iteratively deblend a lightcurve against neighbors

    Parameters
    ----------
    t: array_like
        Time coordinates of lightcurve
    y: array_like
        Brightness measurements (in mag)
    dy: array_like
        Uncertainties for each brightness measurement
    neighbors: list
        List of (t, y, dy) lightcurves for each
        neighbor (NOT including the lightcurve we are deblending)
    period_finding_func: function
        Function used to find the period.  Output format assumed to
        be the same as that used in astrobase.periodbase
    results_storage_container: instance of data_processing.periodsearch_results
        Used to store the results
    which_method: string
        Which period search method is being used
    (optional from here below):
    function_params: dictionary
        A dictionary containing parameters for the function in
        period_finding_func
    nharmonics_fit: int
        Number of harmonics to use in the fit (used to estimate
        flux amplitude)
    nharmonics_resid: int
        Number of harmonics to use to obtain the residual if we
        find that the signal is a blend of a neighboring signal
    ID: string
        ID of the object
    medianfilter: boolean
        whether to median filter the periodogram
    freq_window_epsilon_mf: int
        sets the size of the exclusion area
        in the periodogram for the median filter calculation
    freq_window_epsilon_snr: int
        sets the size of the exclusion area
        in the periodogram for the SNR calculation
    window_size_mf: int
        number of points to include in
        calculating the median value for median filter
    window_size_snr: int
        number of points to include in
        calculating the standard deviation for the SNR
    snr_threshold: float, array_like, or callable
        threshold value or function for
        counting a signal as robust, can be:
             single value -- applies to all objects and periods
             iterable -- length of number of objects, applies
                                each value to each object
             callable -- function of period
    spn_threshold: float or callable
        threshold value or function for
        counting a signal as robust, can be:
             single value -- applies to all objects and periods
             callable -- function of period
    neighbor_peaks_tocheck: int
        when blend is determined, number of neighbor peak periods
        to check that the blended period is actually in the neighbor
    max_blend_recursion: int
        maximum number of blends to try and fit
        out before giving up
    recursion_level: int
        current recursion level
    nworkers: int
        number of child workers
    max_neighbor_amp: float
        the maximum delta F / F0 amplitude a neighbor can have to
        still be counted as a legitimate amplitude; not considered
        for blend checking otherwise
    """

    # Use the period finding function to find the best period
    lsp_dict = period_finding_func(t,y,dy,**function_params)

    # If no period is found at all, quit
    if np.isnan(lsp_dict['bestperiod']):
        if ID:
            print(ID + "\n   -> " + which_method + " found no period, for " + ID)
        else:
            print("   -> " + which_method + " found no period.",flush=True)
        return None

    # Now median filter the periodogram if selected
    if medianfilter:
        pdgm_values = median_filtering(lsp_dict['lspvals'],lsp_dict['periods'],
                                       freq_window_epsilon_mf,
                                       window_size_mf,
                                       t[-1]-t[0],which_method,nworkers=nworkers)

        lsp_dict['medianfilter'] = True
        lsp_dict['lspvalsmf'] = pdgm_values

    # Otherwise just copy periodogram values over
    else:
        pdgm_values = lsp_dict['lspvals']

        # Check that the best period matches what period_finding_func says is best period
        lsp_dict['medianfilter'] = False
        if which_method == 'PDM':
            per_to_comp = lsp_dict['periods'][np.argmin(pdgm_values)]
        elif which_method == 'BLS': # Special handling of inf for BLS
            per_to_comp = lsp_dict['periods'][np.where(np.isinf(pdgm_values),-np.inf,pdgm_values).argmax()]
        else:
            per_to_comp = lsp_dict['periods'][np.argmax(pdgm_values)]
        if abs(per_to_comp - lsp_dict['bestperiod'])/lsp_dict['bestperiod'] > 1e-7:
            print(" Periods: " + str(per_to_comp) + "   " +\
                  str(lsp_dict['bestperiod']))
            raise ValueError("The bestperiod does not match the actual best period w/o median filtering, " + which_method)

    # Get index for the best periodogram value
    if which_method == 'PDM':
        best_pdgm_index = np.argmin(pdgm_values)
    elif which_method == 'BLS': # Special handling of inf for BLS
        best_pdgm_index = np.where(np.isinf(pdgm_values),-np.inf,pdgm_values).argmax()
    else:
        best_pdgm_index = np.argmax(pdgm_values)

    # Set some values
    freq_window_size = freq_window_epsilon_snr/(max(t)-min(t))
    delta_frequency = abs(1./lsp_dict['periods'][1] - 1./lsp_dict['periods'][0])
    freq_window_index_size = int(round(freq_window_size/delta_frequency))

    best_freq = 1./lsp_dict['periods'][best_pdgm_index]


    # Compute periodogram SNR, compare to threshold
    per_snr = snr.periodogram_snr(pdgm_values,lsp_dict['periods'],
                                  best_pdgm_index,
                                  max(t)-min(t),which_method,
                                  freq_window_epsilon=freq_window_epsilon_snr,
                                  rms_window_bin_size=window_size_snr)
    # Print out results
    if ID:
        print("%s\n  %s PERIOD: %.5e days;  pSNR: %.5e"%(ID,which_method,lsp_dict['periods'][best_pdgm_index],per_snr))
    else:
        print("%s PERIOD: %.5e days;  pSNR: %.5e"%(which_method,lsp_dict['periods'][best_pdgm_index],per_snr))

    # Compare to the threshold, if below quit
    if per_snr < snr_threshold_tocomp(snr_threshold,period=lsp_dict['periods'][best_pdgm_index]) or np.isnan(lsp_dict['periods'][best_pdgm_index]):
        if ID:
            print("   -> pSNR not significant enough, for " + ID,flush=True)
        else:
            print("   -> pSNR not significant enough.",flush=True)
        return None

    # Check the signal-to-pink-noise if a threshold is provided
    if spn_threshold is not None:
        if abs(1./lsp_dict['periods'][best_pdgm_index] - 1./lsp_dict['nbestperiods'][0]) >\
                .5*delta_frequency: # If the periods are different, rerun period finding to get stats
            function_params_temp = copy.deepcopy(function_params)
            function_params_temp['startp'] = .999999*lsp_dict['periods'][best_pdgm_index+1]
            function_params_temp['endp'] = lsp_dict['periods'][best_pdgm_index-1]
            lsp_dict_temp = period_finding_func(t,y,dy,**function_params_temp)
            bls_stats_touse = lsp_dict_temp['stats'][0]
            per_temp = lsp_dict_temp['nbestperiods'][0]
        else:
            per_temp = lsp_dict['periods'][best_pdgm_index]
            bls_stats_touse = lsp_dict['stats'][0]
        spn_val = pinknoise.pinknoise_calc(t,y,dy,per_temp,
                                           bls_stats_touse['transitduration'],
                                           bls_stats_touse['transitdepth'],
                                           bls_stats_touse['npoints_in_transit'],
                                           bls_stats_touse['epoch'],
                                           pinknoise.ntransits(t.min(),t.max(),
                                                               bls_stats_touse['epoch'],
                                                               per_temp))
        print("  SPN: %.5e" % spn_val)
        if spn_val < snr_threshold_tocomp(spn_threshold,period=per_temp):
            if ID:
                print("   -> S/PN not significant enough, for " + ID,flush=True)
            else:
                print("   -> S/PN not significant enough.",flush=True)
            return None
    else:
        spn_val=None

    # Check the Baluev FAP is a threshold is provided
    if fap_baluev_threshold is not None:
        fap_baluev_val = fap_baluev(t,dy,lsp_dict['lspvals'][best_pdgm_index],1./lsp_dict['periods'].min())

        print("  B. FAP: %.5e" % fap_baluev_val,flush=True)
        if fap_baluev_val > snr_threshold_tocomp(fap_baluev_threshold,period=lsp_dict['periods'][best_pdgm_index]):
            if ID:
                print("   -> B. FAP too large, for " + ID,flush=True)
            else:
                print("   -> B. FAP too large.",flush=True)
            return None
    else:
        fap_baluev_val = None

    # Fit truncated Fourier series at this frequency
    ff = (FourierFit(nharmonics=nharmonics_fit)
          .fit(t, y, dy, best_freq))
    this_flux_amplitude = ff.flux_amplitude()
    print("THISSSSSSSSSSSSSSSSSSS:")
    _=ff.flux_amplitude(return_df_f0=True)

    # Fit another truncated Fourier series with more harmonics
    ffr = (FourierFit(nharmonics=nharmonics_resid)
           .fit(t, y, dy, best_freq))


    # Now fit Fourier series to all the neighbor light curves
    ffn_all = {}
    for n_ID in neighbors.keys():

        # fit neighbor's lightcurve at this frequency
        ffn = (FourierFit(nharmonics=nharmonics_fit)
               .fit(neighbors[n_ID][0], neighbors[n_ID][1],
                    neighbors[n_ID][2], best_freq))
        ffn_all[n_ID] = ffn

    # Figure out which has maximum amplitude
    max_amp = 0.
    max_ffn_ID = None
    significant_neighbor_blends = []
    toolargeamp_neighbor_blends = []
    for n_ID in ffn_all.keys():
        n_flux_amp, n_df_f = ffn_all[n_ID].flux_amplitude(return_df_f0=True)
        if n_df_f > max_neighbor_amp or np.isnan(n_df_f):
            # Amplitude is too large, don't consider this neighbor
            print("      this neighbor's flux amplitude too large: " + n_ID + "   " + str(n_flux_amp) + "   " + str(n_df_f))
            toolargeamp_neighbor_blends.append(n_ID)
            continue
        if n_flux_amp >\
         results_storage_container.count_neighbor_threshold*this_flux_amplitude:
            significant_neighbor_blends.append(n_ID)
        if n_flux_amp > max_amp:
            max_amp = n_flux_amp
            max_ffn_ID = n_ID



    # If neighbor has larger flux amplitude,
    # then we consider this signal to be a blend.
    # subtract off model signal to get residual
    # lightcurve, and try again
    notmax = False
    if max_ffn_ID:
        print("    checking blends")
        print("    " + max_ffn_ID)
        print("     n: " + str(ffn_all[max_ffn_ID].flux_amplitude()) + " vs.  " + str(this_flux_amplitude),flush=True)
        if ffn_all[max_ffn_ID].flux_amplitude() > this_flux_amplitude:
            if this_flux_amplitude < results_storage_container.stillcount_blend_factor * ffn_all[max_ffn_ID].flux_amplitude():
                # Check that the neighbor actually has this period
                if neighbor_peaks_tocheck > 0:
                    function_params_neighbor = copy.deepcopy(function_params)
                    function_params_neighbor['nbestpeaks'] = neighbor_peaks_tocheck
                    n_lsp_dict = period_finding_func(neighbors[max_ffn_ID][0],neighbors[max_ffn_ID][1],
                                                   neighbors[max_ffn_ID][2],**function_params_neighbor)

                    if not any(np.isclose(n_lsp_dict['nbestperiods'], lsp_dict['periods'][best_pdgm_index], rtol=1e-2, atol=1e-5)):
                        # If the highest-amp blended neighbor doesn't have this period as one of its top periods
                        # Count as a good period
                        print("   -> this isn't a peak period for the neighbor, so ignoring blend.",flush=True)

                        results_storage_container.add_good_period(lsp_dict,t,y,dy,
                                                                  lsp_dict['periods'][best_pdgm_index],
                                              per_snr,
                                              this_flux_amplitude,
                                              significant_neighbor_blends,
                                              ffr.params,
                                              notmax=notmax,
                                              s_pinknoise=spn_val,
                                              fap_baluev=fap_baluev_val,
                                              ignore_blend=max_ffn_ID,
                                              toolargeamp_neighbors=toolargeamp_neighbor_blends)
                        return y - ffr(t)


                print("   -> blended! Trying again.  " + str(ffn_all[max_ffn_ID].flux_amplitude(return_df_f0=True)[1]),flush=True)
                results_storage_container.add_blend(lsp_dict,t,y,dy,max_ffn_ID,
                                                    lsp_dict['periods'][best_pdgm_index],
                                                    per_snr,
                                                    this_flux_amplitude,
                                                    ffn_all[max_ffn_ID].flux_amplitude(),
                                                    ffr.params,
                                                    s_pinknoise=spn_val,
                                                    fap_baluev=fap_baluev_val,
                                                    toolargeamp_neighbors=toolargeamp_neighbor_blends)
                if recursion_level >= max_blend_recursion:
                    print("   Reached the blend recursion level, no longer checking",flush=True)
                    return None
                return iterative_deblend(t, y - ffr(t),
                                         dy, neighbors,
                                         period_finding_func,
                                         results_storage_container,
                                         which_method,
                                         function_params=function_params,
                                         nharmonics_fit=nharmonics_fit,
                                         nharmonics_resid=nharmonics_resid,
                                         ID=ID,
                                         medianfilter=medianfilter,
                                         freq_window_epsilon_mf=freq_window_epsilon_mf,
                                         freq_window_epsilon_snr=freq_window_epsilon_snr,
                                         window_size_mf=window_size_mf,
                                         window_size_snr=window_size_snr,
                                         snr_threshold=snr_threshold,
                                         spn_threshold=spn_threshold,
                                         fap_baluev_threshold=fap_baluev_threshold,
                                         neighbor_peaks_tocheck=neighbor_peaks_tocheck,
                                         max_blend_recursion=max_blend_recursion,
                                         recursion_level=recursion_level+1,
                                         nworkers=nworkers,
                                         max_neighbor_amp=max_neighbor_amp)
            else:
                notmax = True


    # Save the period info and return the pre-whitened light curve
    results_storage_container.add_good_period(lsp_dict,t,y,dy,
                                              lsp_dict['periods'][best_pdgm_index],
                                              per_snr,
                                              this_flux_amplitude,
                                              significant_neighbor_blends,
                                              ffr.params,
                                              notmax=notmax,
                                              s_pinknoise=spn_val,
                                              fap_baluev=fap_baluev_val,
                                              toolargeamp_neighbors=toolargeamp_neighbor_blends)
    return y - ffr(t)
