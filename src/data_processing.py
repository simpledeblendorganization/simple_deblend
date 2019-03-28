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
        lc_objects.__init__(self,radius_)
        if not n_control_workers:
            self.n_control_workers = cpu_count()//4 # Break down over 4 stars in parallel
        elif n_control_workers > cpu_count():
            print("n_control_workers was greater than number of CPUs, setting instead to " + str(cpu_count))
            self.n_control_workers = cpu_count()
        else:
            self.n_control_workers = n_control_workers

        print("n_control_workers is: " + str(self.n_control_workers))


    def run_ls(self,num_periods=3,
               startp=None,endp=None,autofreq=True,
               nbestpeaks=1,periodepsilon=0.1,stepsize=1.0e-4,
               sigclip=float('inf'),nworkers=None,
               verbose=False,medianfilter=False,
               freq_window_epsilon_mf=None,
               freq_window_epsilon_snr=None,
               median_filter_size=None,
               snr_filter_size=None,
               snr_threshold=0.,
               max_blend_recursion=8):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':verbose}
        self.run('LS',ls_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 max_blend_recursion=max_blend_recursion)

        
    def run_pdm(self,num_periods=3,
                startp=None,endp=None,autofreq=True,
                nbestpeaks=1,periodepsilon=0.1,stepsize=1.0e-4,
                sigclip=float('inf'),nworkers=None,
                verbose=False,phasebinsize=0.05,medianfilter=False,
                freq_window_epsilon_mf=None,
                freq_window_epsilon_snr=None,
                median_filter_size=None,
                snr_filter_size=None,
                snr_threshold=0.,
                max_blend_recursion=8):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,
                      'verbose':verbose,'phasebinsize':phasebinsize}

        self.run('PDM',pdm_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 max_blend_recursion=max_blend_recursion)


    def run_bls(self,num_periods=3,
                startp=None,endp=None,autofreq=True,
                nbestpeaks=1,periodepsilon=0.1,stepsize=1.0e-4,
                sigclip=float('inf'),nworkers=None,
                verbose=False,medianfilter=False,
                freq_window_epsilon_mf=None,
                freq_window_epislon_snr=None,
                median_filter_size=None,
                snr_filter_size=None,
                snr_threshold=0.,
                max_blend_recursion=8):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':verbose}

        self.run('BLS',bls_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 max_blend_recursion=max_blend_recursion)
        

    def run(self,which_method,ps_func,params,num_periods,nworkers,
            medianfilter=False,freq_window_epsilon_mf=None,
            freq_window_epsilon_snr=None,median_filter_size=None,
            snr_filter_size=None,snr_threshold=0.,max_blend_recursion=8):

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
        if hasattr(snr_threshold,'__len__'):
            if len(snr_threshold) != len(self.objects):
                raise ValueError("The length of snr_threshold is not the same as the length of objects")
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_val,max_blend_recursion)
                             for o, snr_val in zip(self.objects,snr_threshold)]
        else:
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_threshold,max_blend_recursion)
                             for o in self.objects]

        with ProcessPoolExecutor(max_workers=self.n_control_workers) as executor:
            er = executor.map(self._run_single_object,running_tasks)

        pool_results = [x for x in er]


        #pool_results = []
        #for o in self.objects:
        #    er = self._run_single_object((o,which_method,ps_func,params,
        #                                  num_periods,
        #                                  medianfilter,freq_window_epsilon_mf,
        #                                  freq_window_epsilon_snr,
        #                                  median_filter_size,
        #                                  snr_filter_size,snr_threshold))
        #    pool_results.append(er)



        for result in pool_results:
            if result:
                self.results[result.ID][which_method] = result


    def _run_single_object(self,task):
        (object,which_method,ps_func,params,num_periods,
         medianfilter,freq_window_epsilon_mf,freq_window_epsilon_snr,
         median_filter_size,snr_filter_size,snr_threshold,
         max_blend_recursion) = task

        if not medianfilter:
            if freq_window_epsilon_mf is not None:
                warnings.warn("medianfilter is False, but freq_window_epsilon_mf is not None, not using medianfilter")
            if median_filter_size is not None:
                warnings.warn("medianfilter is False, but median_filter_size is not None, not using median filter")

        neighbor_lightcurves = {neighbor_ID:(self.objects[self.index_dict[neighbor_ID]].times,
                                     self.objects[self.index_dict[neighbor_ID]].mags,
                                     self.objects[self.index_dict[neighbor_ID]].errs) for neighbor_ID in object.neighbors}


        results_storage = periodsearch_results(object.ID)

        yprime = object.mags
        while len(results_storage.good_periods_info) < num_periods:
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
                                           max_blend_recursion=max_blend_recursion)
            if yprime is None:
                #print("yprime is None")
                #print(len(results_storage.good_periods_info))
                break

        if len(results_storage.good_periods_info) > 0 or len(results_storage.blends_info) > 0:
            return results_storage
        else:
            return None

        ###### So what kind of information do I want returned?
          # LSP of any well-found peak
          # Record of which periods (and epochs) were blends
          # Fourier-fitted lc's


    def save_periodsearch_results(self,outputdir):
        if len(results_storage.good_periods_info) > 0:
            with open(self.outputdir + "ps_" + results_storage.ID + ".pkl","wb") as f:
                pickle.dump(results_storage,f)
        

            

class periodsearch_results():
    def __init__(self,ID,count_neighbor_threshold=.25):
        self.ID = ID
        self.good_periods_info = []
        self.blends_info = []
        self.count_neighbor_threshold=count_neighbor_threshold

    def add_good_period(self,lsp_dict,times,mags,errs,snr_value,
                        flux_amplitude,significant_blends):
        dict_to_add = {'lsp_dict':lsp_dict,'times':times,
                       'mags':mags,'errs':errs,
                       'snr_value':snr_value,
                       'flux_amplitude':flux_amplitude,
                       'num_previous_blends':len(self.blends_info),
                       'significant_blends':significant_blends}
        self.good_periods_info.append(dict_to_add)

    def add_blend(self,lsp_dict,times,mags,errs,neighbor_ID,snr_value,
                  flux_amplitude):
        dict_to_add = {'lsp_dict':lsp_dict,
                       'ID_of_blend':neighbor_ID,
                       'snr_value':snr_value,
                       'flux_amplitude':flux_amplitude,
                       'num_previous_signals':len(self.good_periods_info),
                       'times':times,'mags':mags,'errs':errs}
        self.blends_info.append(dict_to_add)


#if __name__ == "__main__":
