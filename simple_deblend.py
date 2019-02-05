from scipy.special import gammaln
import numpy as np
from astropy.stats import LombScargle



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

        yfit = self(phase, in_phase=True)

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
        return X @ self.params






def fap_baluev(t, dy, z, fmax, d_K=3, d_H=1, use_gamma=True):
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

    eZ1 = (z / np.pi) ** 0.5 * (d - 1)
    eZ2 = (1 - z) ** (0.5 * (N_K - 1))

    tau = (g * A / (2 * np.pi)) * eZ1 * eZ2

    Psing = 1 - (1 - z) ** (0.5 * N_K)

    return 1 - Psing * np.exp(-tau)


def iterative_deblend(t, y, dy, neighbors,
                      nharmonics_fit=5,
                      nharmonics_resid=10,
                      max_fap=1e-3,
                      minimum_freq=1./100.,
                      maximum_freq=1./0.1):
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
    nharmonics_fit:
        Number of harmonics to use in the fit (used to estimate
        flux amplitude)
    nharmonics_resid:
        Number of harmonics to use to obtain the residual if we
        find that the signal is a blend of a neighboring signal
    max_fap: float
        Maximum False Alarm Probability (Baluev 2008) to consider.
    minimum_freq: float
        Minimum frequency to search
    maximum_freq: float
        Maximum frequency to search
    """

    # use Lomb-Scargle to get best frequency
    freqs, ls = (LombScargle(t, y, dy)
                 .autopower(minimum_frequency=minimum_freq,
                            maximum_frequency=maximum_freq))

    best_freq = freqs[np.argmax(ls)]

    # compute false alarm probability
    fap = fap_baluev(t, dy, max(ls), maximum_freq)
    print("PERIOD: %.5e days;  FAP: %.5e"%(1./best_freq, fap))

    if fap > max_fap:
        print("  -> not significant enough. No signal found.")
        return None

    # Fit truncated Fourier series at this frequency
    ff = (FourierFit(nharmonics=nharmonics_fit)
          .fit(t, y, dy, best_freq))

    # fit another truncated Fourier series with more harmonics
    ffr = (FourierFit(nharmonics=nharmonics_resid)
           .fit(t, y, dy, best_freq))

    for tn, yn, dyn in neighbors:


        # fit neighbor's lightcurve at this frequency
        ffn = (FourierFit(nharmonics=nharmonics_fit)
               .fit(tn, yn, dyn, best_freq))

        # if neighbor has larger flux amplitude,
        # then we consider this signal to be a blend.
        # subtract off model signal to get residual
        # lightcurve, and try again
        if ffn.flux_amplitude > ff.flux_amplitude:
            print("  -> blended! Trying again.")
            return iterative_deblend(t, y - ffr(t), dy, neighbors,
                                     nharmonics_fit=nharmonics_fit,
                                     nharmonics_resid=nharmonics_resid,
                                     max_fap=max_fap,
                                     minimum_freq=minimum_freq,
                                     maximum_freq=maximum_freq)

    return ff


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

    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.plot(phase, yfit, label='Fit')
    ax.plot(phase, ytrue, label='True')
    ax.errorbar((t * freq) % 1.0, y,
                yerr=dy, capsize=0, color='k', lw=2, fmt='o',
                alpha=0.2)
    ax.legend(loc='best')
    plt.show()

    # Now test the deblending -- we'll feed in the
    # smaller-amplitude lightcurve with its neighboring
    # larger amplitude lightcurve.
    iterative_deblend(tb, yb, dyb, [(t, y, dy)])