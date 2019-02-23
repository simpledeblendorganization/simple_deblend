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
            self.n_control_workers = cpu_count()/4 # Break down over 4 stars in parallel
        elif n_control_workers > cpu_count():
            print("n_control_workers was greater than number of CPUs, setting instead to " + str(cpu_count))
            self.n_control_workers = cpu_count()
        else:
            self.n_control_workers = n_control_workers
        #self._acceptable_methods = ['LS','BLS','PDM']
        self.results = {}


    def run_ls(self,num_periods=3,
                   startp=None,endp=None,autofreq=True,
                   nbestpeaks=3,periodepsilon=0.1,stepsize=1.0e-4,
                   sigclip=float('inf')):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':False}

        self.run('LS',ls_p,params,num_periods)

        
    def run_pdm(self,num_periods=3,
                    startp=None,endp=None,autofreq=True,
                     nbestpeaks=3,periodepsilon=0.1,stepsize=1.0e-4,
                     sigclip=float('inf')):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':False}

        self.run('PDM',pdm_p,params,num_periods)

    def run_bls(self,num_periods=3,
                    startp=None,endp=None,autofreq=True,
                    nbestpeaks=3,periodepsilon=0.1,stepsize=1.0e-4,
                    sigclip=float('inf')):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':False}

        self.run('BLS',bls_p,params,num_periods)
        

    def run(self,which_method,ps_func,params,num_periods):

        with ProcessPoolExecutor(max_workers=self.n_control_workers) as executor:
            _ = executor.map(self._run_single_object,[(o,which_method,ps_func,
                                                       params,num_periods,
                                                       self.n_control_workers)
                                                       for o in self.objects])



    #def _run_single_object(self,object,which_method,ps_func,params,num_periods,
    #                           n_control_workers):
    def _run_single_object(self,task):
        (object,which_method,ps_func,params,num_periods,n_control_workers) = task
        print("Running " + object.ID + "...")
        num_proc_per_run = max(1,cpu_count()/n_control_workers)
        if 'nworkers' in params.keys():
            print("\n***")
            print("params dictionary had nworkers key")
            print("This value is being changed to " + str(num_proc_per_run))
            print("***\n")
        params['nworkers'] = num_proc_per_run
        neighbor_lightcurves = [(self.objects[self.index_dict[neighbor_ID]].times,
                                     self.objects[self.index_dict[neighbor_ID]].mags,
                                     self.objects[self.index_dict[neighbor_ID]].errs) for neighbor_ID in object.neighbors]


        results_storage = periodsearch_results(object.ID)

        while len(results_storage.good_periods_info) < num_periods:
            rv = sdb.iterative_deblend(object.times,object.mags,object.errs,
                                    neighbor_lightcurves,ps_func,
                                    results_storage,
                                    function_params=params,
                                    nharmonics_fit=7,
                                    max_fap=.5)
            if rv is None:
                    break

        if len(results_storage.good_periods_info) > 0:
            if object.ID not in self.results.keys():
                self.results[object.ID] = {}
            self.results[object.ID][which_method] = results_storage

        ###### So what kind of information do I want returned?
          # LSP of any well-found peak
          # Record of which periods (and epochs) were blends
          # Fourier-fitted lc's


    def save_periodsearch_results(self,outputdir):
        for o in self.objects:
            if o.ID in self.results.keys():
                with open(outputdir + "ps_" + o.ID + ".pkl","wb") as f:
                    pickle.dump(self.results[o.ID],f)
        

            

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


if __name__ == "__main__":

    import numpy as np
    
    col_a = lc_collection_for_processing(1.,n_control_workers=4)
    sample_len_1 = 3000
    t1 = np.linspace(0,1200,sample_len_1)
    col_a.add_object(t1,10.+np.sin(t1),[.1]*sample_len_1,0.,0.,'object1')
    col_a.add_object(t1,[10.]*(sample_len_1-1) + [10.0001],[.1]*sample_len_1,0.5,0,'object2')
    col_a.run_ls(startp=6.,endp=7.,stepsize=0.0000001,autofreq=False)
