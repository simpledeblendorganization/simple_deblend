'''data_processing.py - Joshua Wallace - Feb 2019

This code is where the rubber meets the road: where the light curve
data actually gets fed in and results actually get calculated.
'''

from light_curve_class import lc_objects
import simple_deblend
from multiprocessing import Pool, cpu_count
from astrobase.periodbase.zgls import pgen_lsp


class lc_collection_for_processing(lc_objects):
    '''This is the "master class", as it were, of this package.  The methods
    of this class are what actually lead to periods being found, checked,
    calculated, removed, etc. and it is the main way to access the
    abilities of the code

    Subclasses light_curve_class.lc_objects

    The initialization takes one argument and one optional argument:
    radius   - the circular radius, in pixels, for objects to be in 
               sufficient proximity to be regarded as neighbors
    nworkers - (optional; default None) the number of workers to use in the
               parallel calculation
    '''
    def __init__(self,radius_,nworkers_=None):
        lc_objects.__init__(radius_)
        if (not nworkers_) or (nworkers > cpu_count()):
            self.nworkers = cpu_count
        else:
            self.nworkers = nworkers_

class periodsearch_results():
    def __init__(self,ID_):
        self.ID = ID_
        self.good_periods_info = []
        self.blends_info = []

    def add_good_period(self,lsp_dict,times,mags,errs,fap):
        dict_to_add = {'lsp_dict':lsp_dict,'times':times,
                           'mags':mags,'errs':errs,'fap':fap,
                           'num_previous_blends':len(self.blends_info)}
        self.good_periods_info.append(dict_to_add)

    def add_blend(self,lsp_dict,neighbor_ID,fap):
        dict_to_add = {'lsp_dict':lsp_dict,
                           'ID_of_blend':neighbor_ID,
                           'fap':fap,
                           'num_previous_signals':len(self.good_periods_info)}
        self.blend_info.append(dict_to_add)


    def _lomb_scargle_single_object(self,info_passed_in,output_dir):
        object = info_passed_in[0]
        parms = info_passed_in[1]
        # Run GLS on this object
        #initial_lsp = pgen_lsp(object.times,object.mags,object.errs,
        #                           startp=parms['startp'],endp=parms['endp'],
        #                           autofreq=parms['autofreq'],nbestpeaks=1,
        #                           periodepsilon=parms['periodepsilon'],
        #                           stepsize=parms['stepsize'],
        #                           nworkers=1,sigclip=parms['sigclip'],
        #                           verbose=False)

        # Compute FAP on the periods (just best period?)
        # Question for John: what kind of frequency?  Hertz or angular frequency?
        # Question for John: is GLS d_k=1? And what should d_H be?
        #fap = simple_deblend(object.times,object.errs,initial_lsp['lspvals'],
        #                         1/initial_lsp['startp'],d_K=1)
                                                                

        # Fit Fourier series to get flux amplitude

        # Fit Fourier series for all neighbors to determine their
        # flux amplitudes


        # Compare flux amplitudes


        # If neighbor has larger flux amplitude,
        # mark it as such and rerun with the residual LC



        ####
        # Actually, just try John's function
        neighbor_lightcurves = [(self.lc_objects.objects[self.lc_objects.objects.index_dict[neighbor_ID]].times,
                                     self.lc_objects.objects[self.lc_objects.objects.index_dict[neighbor_ID]].mags,
                                     self.lc_objects.objects[self.lc_objects.objects.index_dict[neighbor_ID]].errs) for neighbor_ID in object.neighbors]

        results_storage = periodsearch_results(object.ID)

        for _ in range(parms['nbestpeaks']):
            rv = iterative_deblend(object.times,object.mags,object.errs,
                                    neighbor_lightcurves,pgen,
                                    results_storage,
                                    function_parms=parms,
                                    nharmonics_fit=7,
                                    max_fap=.5,minimum_freq=1./parms['endp'],
                                    maximum_freq=1./parms['startp'])
            if rv is None:
                    break

        if len(results_storage.good_periods_info) > 0:
            with open(output_dir + "periodsearch_" + object.ID + ".pkl") as f:
                pickle.dump(results_storage,f)

        ###### So what kind of information do I want returned?
          # LSP of any well-found peak
          # Record of which periods (and epochs) were blends
          # Fourier-fitted lc's
        

    def lomb_scargle_run(self,startp=None,endp=None,autofreq=True,
                             nbestpeaks=3,periodepsilon=0.1,stepsize=1.0e-4,
                             sigclip=float('inf'),output_dir="."):

        params = {startp:startp,endp:endp,autofreq:autofreq,
                      nbestpeaks:nbestpeaks,periodepsilon:periodepsilon,
                      stepsize:stepsize,sigclip:sigclip}

        
        mp_pool = Pool(self.nworkers)

        _ = pool.map(_lomb_scargle_single_object, [(o,params,output_dir)
                                                       for o in self.objects])
        
