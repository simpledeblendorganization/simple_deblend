'''data_processing.py - Joshua Wallace - Feb 2019

This code is where the rubber meets the road: where the light curve
data actually gets fed in and results actually get calculated.
'''

from light_curve_class import lc_objects
import simple_deblend as sdb
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from astrobase.periodbase.zgls import pgen_lsp as ls_p
from astrobase.periodbase.spdm import stellingwerf_pdm as pdm_p
from astrobase.periodbase.kbls import bls_parallel_pfind as bls_p

import copy
import pickle
import warnings

from os.path import isfile



class lc_collection_for_processing(lc_objects):
    '''This is the "master class", as it were, of this package.  The methods
    of this class are what actually lead to periods being found, checked,
    calculated, removed, etc. and it is the main way to access the
    abilities of the code

    Subclasses light_curve_class.lc_objects

    The initialization takes one argument and one optional argument:
    radius   - the circular radius, in pixels, for objects to be in 
               sufficient proximity to be regarded as neighbors
    n_control_workers - (optional; default None) the number of workers to use in the
               parallel calculation---value of None defaults to 
               multiprocessing.cpu_count
    '''

    def __init__(self,radius_,n_control_workers=None):

        # Initialize the collection of light curves
        lc_objects.__init__(self,radius_)

        # Figure out how many n_control_workers to use
        if not n_control_workers:
            self.n_control_workers = cpu_count()//4 # Default, break down over 4 stars in parallel
        elif n_control_workers > cpu_count():
            print("n_control_workers was greater than number of CPUs, setting instead to " + str(cpu_count))
            self.n_control_workers = cpu_count()
        else:
            self.n_control_workers = n_control_workers

        print("n_control_workers is: " + str(self.n_control_workers))




    def run_ls(self,num_periods=3,
               startp=None,endp=None,autofreq=True,
               nbestpeaks=1,periodepsilon=0.1,stepsize=None,
               sigclip=float('inf'),nworkers=None,
               verbose=False,medianfilter=False,
               freq_window_epsilon_mf=None,
               freq_window_epsilon_snr=None,
               median_filter_size=None,
               snr_filter_size=None,
               snr_threshold=0.,
               fap_baluev_threshold=0.,
               max_blend_recursion=4,
               outputdir="."):
        '''Run a Lomb-Scargle period search

        This takes a number of optional arguments:
        num_periods           - maximum number of periods to search for

        startp                - minimum period of the search

        endp                  - maximum period of the search

        autofreq              - astrobase autofreq parameter, whether to
               automatically determine the frequency grid

        nbestpeaks            - astrobase nbestpeaks parameter, right now
               adjusting this shouldn't change the code at all

        periodepsilon         - astrobase periodepsilon parameter

        stepsize              - astrobase stepsize parameter, if setting
               manual frequency grid

        sigclip               - astrobase sigclip parameter, sigma
               clipping light curve

        nworkers              - astrobase nworkers parameter, None
               value leads to automatic determination

        verbose               - astrobase verbose parameter

        medianfilter          - whether to median filter the periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        median_filter_size    - number of points to include in 
               calculating the median value for median filter

        snr_filter_size       - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust from periodogram SNR, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        fap_baluev_threshold  - threshold value or function for
               counting a signal as robust from sdb.fap_baluev(), can be:
                    single value -- applies to all objects and periods
                    callable -- function of period

        max_blend_recursion   - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''

        # Value checking
        if autofreq and stepsize:
            raise ValueError("autofreq was set to True, but stepsize was given")

        # Set up params dict for the astrobase search
        if autofreq:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'sigclip':sigclip,'verbose':verbose}
        else:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':verbose}
        
        # The period search method
        method = 'LS'

        # Call run 
        self.run(method,ls_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 fap_baluev_threshold=fap_baluev_threshold,
                 max_blend_recursion=max_blend_recursion,
                 outputdir=outputdir)

        
    def run_pdm(self,num_periods=3,
                startp=None,endp=None,autofreq=True,
                nbestpeaks=1,periodepsilon=0.1,stepsize=None,
                sigclip=float('inf'),nworkers=None,
                verbose=False,phasebinsize=0.05,medianfilter=False,
                freq_window_epsilon_mf=None,
                freq_window_epsilon_snr=None,
                median_filter_size=None,
                snr_filter_size=None,
                snr_threshold=0.,
                max_blend_recursion=4,
                outputdir="."):
        '''Run a Phase Dispersion Minimization period search

        This takes a number of optional arguments:
        num_periods           - maximum number of periods to search for

        startp                - minimum period of the search

        endp                  - maximum period of the search

        autofreq              - astrobase autofreq parameter, whether to
               automatically determine the frequency grid

        nbestpeaks            - astrobase nbestpeaks parameter, right now
               adjusting this shouldn't change the code at all

        periodepsilon         - astrobase periodepsilon parameter

        stepsize              - astrobase stepsize parameter, if setting
               manual frequency grid

        sigclip               - astrobase sigclip parameter, sigma
               clipping light curve

        nworkers              - astrobase nworkers parameter, None
               value leads to automatic determination

        verbose               - astrobase verbose parameter

        phasebinsize          - astrobase phasebinsize parameter

        medianfilter          - whether to median filter the periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        median_filter_size    - number of points to include in 
               calculating the median value for median filter

        snr_filter_size       - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        max_blend_recursion   - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''

        # Value checking
        if autofreq and stepsize:
            raise ValueError("autofreq was set to True, but stepsize was given")

        # Set up params dict for the astrobase search
        if autofreq:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'sigclip':sigclip,
                      'verbose':verbose,'phasebinsize':phasebinsize}
        else:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,
                      'verbose':verbose,'phasebinsize':phasebinsize}

        # The period search method
        method = 'PDM'
        
        # Call run
        self.run(method,pdm_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 max_blend_recursion=max_blend_recursion,
                 outputdir=outputdir)


    def run_bls(self,num_periods=3,
                startp=None,endp=None,autofreq=True,
                nbestpeaks=1,periodepsilon=0.1,
                nphasebins=None,stepsize=None,
                mintransitduration=0.01,maxtransitduration=0.4,
                sigclip=float('inf'),nworkers=None,
                verbose=False,medianfilter=False,
                freq_window_epsilon_mf=None,
                freq_window_epsilon_snr=None,
                median_filter_size=None,
                snr_filter_size=None,
                snr_threshold=0.,
                spn_threshold=0.,
                max_blend_recursion=3,
                outputdir="."):
        '''Run a Box-fitting Least Squares period search

        This takes a number of optional arguments:
        num_periods           - maximum number of periods to search for

        startp                - minimum period of the search

        endp                  - maximum period of the search

        autofreq              - astrobase autofreq parameter, whether to
               automatically determine the frequency grid

        nbestpeaks            - astrobase nbestpeaks parameter, right now
               adjusting this shouldn't change the code at all

        periodepsilon         - astrobase periodepsilon parameter

        nphasebins            - astrobase nphasebins parameter

        stepsize              - astrobase stepsize parameter, if setting
               manual frequency grid

        mintransitduration    - astrobase mintransitduration parameter,
               the minimum transit duration to search

        maxtransitduration    - astrobase maxtransitduration parameter,
               the maximum transit duration to search

        sigclip               - astrobase sigclip parameter, sigma
               clipping light curve

        nworkers              - astrobase nworkers parameter, None
               value leads to automatic determination

        verbose               - astrobase verbose parameter

        phasebinsize          - astrobase phasebinsize parameter

        medianfilter          - whether to median filter the periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        median_filter_size    - number of points to include in 
               calculating the median value for median filter

        snr_filter_size       - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust, in the periodogram, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        spn_threshold         - threshold value or function for
               counting signal-to-pink-noise as robust, can be:
                    single value -- applies to all objects and periods
                    callable -- function of period

        max_blend_recursion   - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''

        # Value checking
        if (autofreq and nphasebins) or (autofreq and stepsize):
            raise ValueError("autofreq was set to true, but stepsize and/or nphasebins was given")

        # Set params dict for astrobase
        if autofreq:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'mintransitduration':mintransitduration,
                      'maxtransitduration':maxtransitduration,
                      'sigclip':sigclip,'verbose':verbose}
        else:
            if not stepsize:
                raise ValueError("autofreq is false, but no value given for stepsize")
            if not nphasebins:
                raise ValueError("autofreq is false, but no value given for nphasebins")
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'mintransitduration':mintransitduration,
                      'maxtransitduration':maxtransitduration,
                      'stepsize':stepsize,'nphasebins':nphasebins,
                      'sigclip':sigclip,'verbose':verbose}

        # The period search method
        method = 'BLS'

        # Call run
        self.run(method,bls_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 spn_threshold=spn_threshold,
                 max_blend_recursion=max_blend_recursion,
                 outputdir=outputdir)
        

    def run(self,which_method,ps_func,params,num_periods,nworkers,
            medianfilter=False,freq_window_epsilon_mf=None,
            freq_window_epsilon_snr=None,median_filter_size=None,
            snr_filter_size=None,snr_threshold=0.,spn_threshold=None,
            fap_baluev_threshold=None,
            max_blend_recursion=8,outputdir="."):
        '''Run a given period search method

        which_method  - the name of the period search method being used
        ps_func       - the period search function from astrobase
        params        - params dict to be passed to ps_func
        num_periods   - maximum number of periods to search
        nworkers      - number of child workers per control worker,
                        can be automatically determined

        Optional parameters:

        medianfilter   - whether to perform median filtering of periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        median_filter_size - number of points to include in 
               calculating the median value for median filter

        snr_filter_size    - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        spn_threshold         - threshold value or function for
               counting signal-to-pink-noise as robust, can be:
                    single value -- applies to all objects and periods
                    callable -- function of period
                    None     -- ignore SPN calculation entirely

        fap_baluev_threshold  - threshold value or function for
               counting Baluev FAP measure as robust, can be:
                    single value -- applies to all objects and periods
                    callable -- function of period
                    None     -- ignore fap_baluev calculation entirely

        max_blend_recursion - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''


        # Set nworkers
        num_proc_per_run = max(1,cpu_count()//self.n_control_workers)
        if nworkers is None:
            print("\n***")
            print("None value given to nworkers, auto-calculating a value")
            print("\n***")
        else:
            print("\n***")
            print("nworkers value given")
            if nworkers > num_proc_per_run:
                print("Its value, " + str(nworkers) + " is too large given")
                print(" the number of CPUs and number of control processes")
                print("It is being changed to " + str(num_proc_per_run))

            else:
                num_proc_per_run = nworkers
            print("***\n")
        print("Number of worker processes per control process: " + str(num_proc_per_run) + "\n")

        params['nworkers'] = num_proc_per_run

        # Check what snr_threshold is and set up the tasks list accordingly
        if hasattr(snr_threshold,'__len__'):
            if len(snr_threshold) != len(self.objects):
                raise ValueError("The length of snr_threshold is not the same as the length of objects")
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_val,spn_threshold,
                              fap_baluev_threshold,
                              max_blend_recursion,
                              num_proc_per_run,outputdir)
                             for o, snr_val in zip(self.objects,snr_threshold)]
        elif callable(snr_threshold): # If a callable thing of some kind
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_threshold,spn_threshold,
                              fap_baluev_threshold,
                              max_blend_recursion,
                              num_proc_per_run,outputdir)
                             for o in self.objects]
        else:
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_threshold,spn_threshold,
                              fap_baluev_threshold,
                              max_blend_recursion,
                              num_proc_per_run,outputdir)
                             for o in self.objects]



        # Start the run
        print("**************************************")
        print("******")
        print("******       Starting " + which_method + " run")
        print("******")
        print("**************************************")
        with ProcessPoolExecutor(max_workers=self.n_control_workers) as executor:
            er = executor.map(self._run_single_object,running_tasks)

        # Collect the results
        pool_results = [x for x in er]

        for result in pool_results:
            if result:
                self.results[result.ID][which_method] = result


    def _run_single_object(self,task):
        ''' Used to run the code for just a single object,
        included in this way to make the code parallelizable.

        task - the task passed from self.run, see that method
               for definition
        '''

        # Extract parameters from task
        (object,which_method,ps_func,params,num_periods,
         medianfilter,freq_window_epsilon_mf,freq_window_epsilon_snr,
         median_filter_size,snr_filter_size,snr_threshold,spn_threshold,
         fap_baluev_threshold,
         max_blend_recursion,nworkers,outputdir) = task

        # Value checking
        if not medianfilter:
            if freq_window_epsilon_mf is not None:
                warnings.warn("medianfilter is False, but freq_window_epsilon_mf is not None, not using medianfilter")
            if median_filter_size is not None:
                warnings.warn("medianfilter is False, but median_filter_size is not None, not using median filter")

        if spn_threshold and which_method.lower() != 'bls':
            raise ValueError("spn_threshold has a value, but the method is not BLS")

        if fap_baluev_threshold and which_method.lower() != 'ls':
            raise ValueError("fap_baluev has a value, but the method is not LS,")


        # Collect the neighbor light curves
        neighbor_lightcurves = {neighbor_ID:(self.objects[self.index_dict[neighbor_ID]].times,
                                     self.objects[self.index_dict[neighbor_ID]].mags,
                                     self.objects[self.index_dict[neighbor_ID]].errs) for neighbor_ID in object.neighbors}

        # Create a place to store the results
        results_storage = periodsearch_results(object.ID)

        # Start iterating
        yprime = object.mags
        while len(results_storage.good_periods_info) < num_periods:
            # Try the iterative deblend
            yprime = sdb.iterative_deblend(object.times,yprime,object.errs,
                                           neighbor_lightcurves,ps_func,
                                           results_storage,
                                           which_method,
                                           function_params=params,
                                           nharmonics_fit=7,
                                           ID=str(object.ID),
                                           medianfilter=medianfilter,
                                           freq_window_epsilon_mf=freq_window_epsilon_mf,
                                           freq_window_epsilon_snr=freq_window_epsilon_snr,
                                           window_size_mf=median_filter_size,
                                           window_size_snr=snr_filter_size,
                                           snr_threshold=snr_threshold,
                                           spn_threshold=spn_threshold,
                                           fap_baluev_threshold=fap_baluev_threshold,
                                           max_blend_recursion=max_blend_recursion,
                                           nworkers=nworkers)
            if yprime is None: # No more results to be had
                break

        # Save as we go
        with open(outputdir + "/ps_" + str(object.ID) + "_" +\
                      which_method + "_goodperiod.pkl","wb") as f:
            pickle.dump(results_storage.good_periods_info,f)
        with open(outputdir + "/ps_" + str(object.ID) + "_" +\
                      which_method + "_blends.pkl","wb") as f:
            pickle.dump(results_storage.blends_info,f)
        if len(results_storage.good_periods_info) > 0 or\
            len(results_storage.blends_info) > 0:
            # And return
            return results_storage

        else:
            return None



    def save_periodsearch_results(self,outputdir):
        '''Method used to save the results

        outputdir  - the directory to which to save the results
        '''
        
        print("Saving results...")

        # Loop over the objects
        for k in self.results.keys():
            # Loop over the period search methods
            for k2 in self.results[k].keys():
                r = self.results[k][k2]
                with open(outputdir + "/ps_" + r.ID + "_" + k2 + "_goodperiod.pkl","wb") as f:
                        pickle.dump(r.good_periods_info,f)
                with open(outputdir + "/ps_" + r.ID + "_" + k2 + "_blends.pkl","wb") as f:
                        pickle.dump(r.blends_info,f)


          

            

class periodsearch_results():
    '''A container to store the results of the above
    period search

    The initialization takes one argument and two optional arguments:
    ID                       - ID of object being stored

    count_neighbor_threshold - flux amplitude ratio needs to be at least
                   this for an object to store a blended neighbor's info

    stillcount_blend_factor - flux amplitude ratio needs to be at least this
                   when the object has a lower amplitude for this to 
                   still count as a period for the object
    '''

    def __init__(self,ID,count_neighbor_threshold=0.25,
                 stillcount_blend_factor=0.9):
        self.ID = ID
        self.good_periods_info = []
        self.blends_info = []
        self.count_neighbor_threshold=count_neighbor_threshold
        self.stillcount_blend_factor=stillcount_blend_factor


    def add_good_period(self,lsp_dict,times,mags,errs,period,snr_value,
                        flux_amplitude,significant_blends,
                        ff_params,
                        notmax=False,s_pinknoise=None,
                        fap_baluev=None,
                        ignore_blend=False,
                        toolargeamp_neighbors=[]):
        '''add a good period for the object

        lsp_dict   - the astrobase lsp_dict
        times      - light curve times
        mags       - light curve magnitudes
        errs       - light curve errors
        period     - the associated period
        snr_value  - value of the periodogram SNR
        flux_amplitude - flux amplitude
        significant_blends - neighbors with flux amplitudes above
                             self.count_neighbor_threshold
        ffparams - Fourier fit parameters for the *current* LC
        notmax     - if the object does not have the maximum
                     flux amplitude but is greater than
                     self.stillcount_blend_factor
        s_pinknoise -signal to pink noise value, only for BLS
        ignore_blend - ID of blend being ignore if its being
                       ignored, False otherwise
        toolargeamp_neighbors - list of IDs of neighbors with
                       too large amplitudes to be counted
        '''
        dict_to_add = {'lsp_dict':lsp_dict,
                       'period':period,'times':times,
                       'mags':mags,'errs':errs,
                       'snr_value':snr_value,
                       'flux_amplitude':flux_amplitude,
                       'num_previous_blends':len(self.blends_info),
                       'significant_blends':significant_blends,
                       'not_max':notmax,'ignore_blend':ignore_blend,
                       'ff_params':ff_params}
        if s_pinknoise is not None:
            dict_to_add['s_pinknoise'] = s_pinknoise
        if fap_baluev is not None:
            dict_to_add['fap_baluev_log'] = fap_baluev
        if toolargeamp_neighbors:
            dict_to_add['toolargeamp_neighbors'] = toolargeamp_neighbors
        self.good_periods_info.append(dict_to_add)

    def add_blend(self,lsp_dict,times,mags,errs,neighbor_ID,
                  period,snr_value,
                  flux_amplitude,neighbor_flux_amplitude,
                  ff_params,s_pinknoise=None,
                  fap_baluev=None,toolargeamp_neighbors=[]):
        '''add info where the object is blended with another object,
        that object being determined as the variability source

        lsp_dict   - the astrobase lsp_dict
        times      - light curve times
        mags       - light curve magnitudes
        errs       - light curve errors
        neighbor_ID - ID of the variability source
        period     - the associated period
        snr_value  - value of the periodogram SNR
        flux_amplitude - flux amplitude
        neighbor_flux_amplitude - flux amplitude of blended neighbor
        ffparams - Fourier fit parameters for the *current* LC
        s_pinknoise -signal to pink noise value, only for BLS
        toolargeamp_neighbors - list of IDs of neighbors with
                       too large amplitudes to be counted
        '''
        dict_to_add = {'lsp_dict':lsp_dict,
                       'ID_of_blend':neighbor_ID,
                       'period':period,
                       'snr_value':snr_value,
                       'flux_amplitude':flux_amplitude,
                       'neigh_flux_amplitude':neighbor_flux_amplitude,
                       'num_previous_signals':len(self.good_periods_info),
                       'times':times,'mags':mags,'errs':errs,
                       'ff_params':ff_params}
        if s_pinknoise is not None:
            dict_to_add['s_pinknoise'] = s_pinknoise
        if fap_baluev is not None:
            dict_to_add['fap_baluev_log'] = fap_baluev
        if toolargeamp_neighbors:
            dict_to_add['toolargeamp_neighbors'] = toolargeamp_neighbors
        self.blends_info.append(dict_to_add)


