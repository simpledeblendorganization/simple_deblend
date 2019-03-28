from scipy.stats import sigmaclip
import numpy as np
import math
import snr_calculation as snr



def _2d(arr):
    """ Reshape 1d array to 2d array """
    return np.reshape(arr, (len(arr), 1))


def design_matrix(t, freq, nharms):
    """
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
    """

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
    def __init__(self, nharmonics=5):
        self.nharmonics = nharmonics


    def fit(self, t, y, dy, freq, nharmonics=None,
            dy_as_weights=False,
            y_as_flux=False):

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

        self.params = np.linalg.solve(S, z)

        return self

    @property
    def flux_amplitude(self):

        # delta flux / f0

        phase = np.linspace(0, 1, 200)

        yfit = self(phase, in_phase=True) + self.params[0]

        if self.y_as_flux:
            return 0.5 * (max(yfit) - min(yfit))

        mbrite = min(yfit)
        mdim = max(yfit)

        return 10 ** (-0.4 * mbrite) - 10 ** (-0.4 * mdim)


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
        freq = 1 if in_phase else self.freq
        X = design_matrix(t, freq, self.nharmonics)
        return X @ self.params - self.params[0]



def median_filtering(lspvals,periods,freq_window_epsilon,median_filter_size,duration,which_method):
    # First, make sure all the frequency values are equally spaced
    diff = np.diff(1./periods)
    for val in diff:
        if not math.isclose(val,diff[0],rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError("The frequency differences are not equal spacing, which this function assumes")

    freq_window_size = freq_window_epsilon/duration
    freq_window_index_size = int(round(freq_window_size/abs(1./periods[0] - 1./periods[1])))


    median_filter_values = []
    for i in range(len(lspvals)):
        window_vals = []
        if i >= freq_window_index_size:
            if i - freq_window_index_size < 0:
                raise RuntimeError("Too small, " + str(i-freq_window_index_size))
            window_vals.extend(lspvals[max(0,i-freq_window_index_size-median_filter_size+1):i-freq_window_index_size+1].tolist())
        if i + freq_window_index_size < len(lspvals):
            window_vals.extend(lspvals[i+freq_window_index_size:i+freq_window_index_size+median_filter_size].tolist())
        window_vals = np.array(window_vals)
        wherefinite = np.isfinite(window_vals)
        vals, low, upp = sigmaclip(window_vals[wherefinite],low=3,high=3)

        median_filter_values.append(np.median(vals))

    #plt.plot(periods,lspvals,color='red',lw=.9)
    #plt.plot(periods,lspvals - median_filter_values+1.,color='blue',lw=.5)
    #plt.savefig("temp.pdf")
    #quit()

    if which_method == 'PDM':
        return lspvals + (1. - np.array(median_filter_values))
    else:
        return lspvals - np.array(median_filter_values)


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
                      max_blend_recursion=8,
                      recursion_level=0):
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
    function_params: dictionary
        A dictionary containing parameters for the function in
        period_finding_func
    nharmonics_fit:
        Number of harmonics to use in the fit (used to estimate
        flux amplitude)
    nharmonics_resid:
        Number of harmonics to use to obtain the residual if we
        find that the signal is a blend of a neighboring signal
    """

    # use the function to find the best period
    lsp_dict = period_finding_func(t,y,dy,**function_params)

    if np.isnan(lsp_dict['bestperiod']):
        if ID:
            print(ID + "\n   -> period finding function found no period, for " + ID)
        else:
            print("   -> period finding function found no period.")
        return None

    # Now median filter the periodogram if selected
    if medianfilter:
              
        pdgm_values = median_filtering(lsp_dict['lspvals'],lsp_dict['periods'],
                                       freq_window_epsilon_mf,
                                       window_size_mf,
                                       t[-1]-t[0],which_method)

        lsp_dict['medianfilter'] = True
        lsp_dict['lspvalsmf'] = pdgm_values

    else:
        pdgm_values = lsp_dict['lspvals']

        lsp_dict['medianfilter'] = False
        if which_method == 'PDM':
            per_to_comp = lsp_dict['periods'][np.argmin(pdgm_values)]
        else:
            per_to_comp = lsp_dict['periods'][np.argmax(pdgm_values)]
        if abs(per_to_comp - lsp_dict['bestperiod'])/lsp_dict['bestperiod'] > 1e-7:
            print(" Periods: " + str(per_to_comp) + "   " +\
                  str(lsp_dict['bestperiod']))
            raise ValueError("The bestperiod does not match the actual best period w/o median filtering")

    if which_method == 'PDM':
        best_pdgm_index = np.argmin(pdgm_values)
    else:
        best_pdgm_index = np.argmax(pdgm_values)

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
    if ID:
        print("%s\n PERIOD: %.5e days;  pSNR: %.5e"%(ID,lsp_dict['periods'][best_pdgm_index],per_snr))
    else:
        print("PERIOD: %.5e days;  pSNR: %.5e"%(lsp_dict['periods'][best_pdgm_index],per_snr))

    if per_snr < snr_threshold or np.isnan(lsp_dict['periods'][best_pdgm_index]):
        if ID:
            print("   -> not significant enough, for " + ID)
        else:
            print("   -> not significant enough.")
        return None

    # Fit truncated Fourier series at this frequency
    ff = (FourierFit(nharmonics=nharmonics_fit)
          .fit(t, y, dy, best_freq))
    this_flux_amplitude = ff.flux_amplitude

    # fit another truncated Fourier series with more harmonics
    ffr = (FourierFit(nharmonics=nharmonics_resid)
           .fit(t, y, dy, best_freq)) 



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
    for n_ID in ffn_all.keys():
        if ffn_all[n_ID].flux_amplitude >\
         results_storage_container.count_neighbor_threshold*this_flux_amplitude:
            significant_neighbor_blends.append(n_ID)
        if ffn_all[n_ID].flux_amplitude > max_amp:
            max_amp = ffn_all[n_ID].flux_amplitude
            max_ffn_ID = n_ID

        

    # if neighbor has larger flux amplitude,
    # then we consider this signal to be a blend.
    # subtract off model signal to get residual
    # lightcurve, and try again
    if max_ffn_ID:
        print("    checking blends")
        print("    " + max_ffn_ID)
        print("     n: " + str(ffn_all[max_ffn_ID].flux_amplitude) + " vs.  " + str(this_flux_amplitude))
        if ffn_all[max_ffn_ID].flux_amplitude > this_flux_amplitude: 
            print("   -> blended! Trying again.")
            results_storage_container.add_blend(lsp_dict,t,y,dy,max_ffn_ID,snr_threshold,
                                                this_flux_amplitude)
            if recursion_level >= max_blend_recursion:
                print("   Reached the blend recursion level, no longer checking")
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
                                     max_blend_recursion=max_blend_recursion,
                                     recursion_level=recursion_level+1)


    # Return the period and the pre-whitened light curve
    
    results_storage_container.add_good_period(lsp_dict,t,y,dy,
                                              snr_threshold,this_flux_amplitude,
                                              significant_neighbor_blends)
    return y - ffr(t)


if __name__ == '__main__':
    def fake_lc(c, s, n=100, freq=1, sigma=0.1, baseline=1):
        rand = np.random.RandomState(100)

        t = baseline * 365 * np.sort(rand.rand(n))
        y = np.zeros_like(t)
        for h, (C, S) in enumerate(zip(c, s)):
            phi = 2 * np.pi * (h+1) * t * freq

            y += C * np.cos(phi) + S * np.sin(phi)

        y += sigma * rand.randn(n)
        dy = sigma * np.ones_like(y)

        return t, y, dy

    # example signal
    c = np.array([0.1, 0.2, 0.1])
    s = np.array([0.2, 0.1, 0.])
    freq=1


    # create the original lightcurve
    t, y, dy = fake_lc(c, s, freq=freq)

    # now create a neighbor with the same signal but smaller amplitude
    # (i.e., the blend)
    tb, yb, dyb = fake_lc(c * 0.5, s * 0.5, freq=freq)

    # visualize the signal to check that the Fourier fitting
    # is doing OK
    ff = FourierFit().fit(t, y, dy, freq)

    ff_true = FourierFit.from_params(c, s, 0, freq)

    phase = np.linspace(0, 1, 200)
    yfit = ff(phase, in_phase=True)
    ytrue = ff_true(phase, in_phase=True)

    #import matplotlib.pyplot as plt
    #f, ax = plt.subplots()
    #ax.plot(phase, yfit, label='Fit')
    #ax.plot(phase, ytrue, label='True')
    #ax.errorbar((t * freq) % 1.0, y,
    #            yerr=dy, capsize=0, color='k', lw=2, fmt='o',
    #            alpha=0.2)
    #ax.legend(loc='best')
    #plt.show()

    # Now test the deblending -- we'll feed in the
    # smaller-amplitude lightcurve with its neighboring
    # larger amplitude lightcurve.
    iterative_deblend(tb, yb, dyb, [(t, y, dy)])
