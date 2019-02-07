'''data_processing.py - Joshua Wallace - Feb 2019

This code is where the rubber meets the road: where the light curve
data actually gets fed in and results actually get calculated.
'''

import light_curve_class.lc_objects as lc_objects
from multiprocessing import Pool, cpu_count
from astrobase.periodbase.zgls import pgen_lsp


class lc_collection_for_processing(lc_objects):
    def __init__(self,radius_,nworkers_=None):
        lc_objects.__init__(radius_)
        if (not nworkers_) or (nworkers > cpu_count()):
            self.nworkers = cpu_count
        else:
            self.nworkers = nworkers_


    def _lomb_scargle_single_object(self,info_passed_in):
        object = info_passed_in[0]
        parms = info_passed_in[1]
        # Run GLS on this object
        initial_lsp = pgen_lsp(object.times,object.mags,object.errs,
                                   startp=parms['startp'],endp=parms['endp'],
                                   autofreq=parms['autofreq'],nbestpeaks=1,
                                   periodepsilon=parms['periodepsilon'],
                                   stepsize=parms['stepsize'],
                                   nworkers=1,sigclip=parms['sigclip'],
                                   verbose=False)

        # Compute FAP on the periods (just best period?)

        # Fit Fourier series to get flux amplitude

        # Fit Fourier series for all neighbors to determine their
        # flux amplitudes


        # Compare flux amplitudes


        # If neighbor has larger flux amplitude,
        # mark it as such and rerun with the residual LC


        ###### So what kind of information do I want returned?
          # LSP of any well-found peak
          # Record of which periods (and epochs) were blends
        

    def lomb_scargle_run(self,startp=None,endp=None,autofreq=True,
        nbestpeaks=3,periodepsilon=0.1,stepsize=1.0e-4,sigclip=float(inf)):

        params = {startp=startp,endp=endp,autofreq=autofreq,
                      nbestpeaks=nbestpeaks,periodepsilon=periodepsilon,
                      stepsize=stepsize,sigclip=sigclip}

        
        mp_pool = Pool(self.nworkers)

        _ = pool.map(_lomb_scargle_single_object, [(o,params) for o in self.objects])
        
