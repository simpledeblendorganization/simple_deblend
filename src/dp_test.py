from light_curve_class import lc_objects
from multiprocessing import Pool, cpu_count
import numpy as np







def iterative_deblend(t, y, dy, depth, max_depth,
                      function_params=None,
                      nharmonics_fit=5,
                      nharmonics_resid=10,
                      max_fap=1e-3):




    # compute false alarm probability
    best_freq = 1./(2.*np.pi)





    # if neighbor has larger flux amplitude,
    # then we consider this signal to be a blend.
    # subtract off model signal to get residual
    # lightcurve, and try again
    if depth < max_depth: # Need to look at ambiguous cases, also need to find which one is likely blend source, not just first blend source
        return iterative_deblend(t, y - .01, dy, depth+1, max_depth,
                                 function_params=function_params,
                                 nharmonics_fit=nharmonics_fit,
                                 nharmonics_resid=nharmonics_resid,
                                 max_fap=max_fap)

    #return ff
    # Return the period and the pre-whitened light curve
    return "hi"




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
               parallel calculation---value of None defaults to 
               multiprocessing.cpu_count
    '''
    def __init__(self,radius_,nworkers=None):
        lc_objects.__init__(self,radius_)
        if not nworkers:
            self.nworkers = cpu_count
        elif nworkers > cpu_count():
            print("nworkers was greater than number of CPUs, setting instead to " + str(cpu_count))
            self.nworkers = cpu_count
        else:
            self.nworkers = nworkers
        #self._acceptable_methods = ['LS','BLS','PDM']
        self.results = {}


    def run_ls(self,num_periods=3,
                   startp=None,endp=None,autofreq=True,
                   nbestpeaks=3,periodepsilon=0.1,stepsize=1.0e-4,
                   sigclip=float('inf')):

        params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':False}

        self.run('LS',params,num_periods)

        
        

    def run(self,which_method,params,num_periods):

        mp_pool = Pool(self.nworkers)

        #_ = mp_pool.map(self._run_single_object, [{'object':o,
        #                                           'which_method':which_method,
        #                                           'ps_func':ps_func,
        #                                            'params':params,
        #                                            'num_periods':num_periods}
        #                                               for o in self.objects])
        for o in self.objects:
            print("******\n" + o.ID + "\n******\n")
            self._run_single_object(o,which_method,
                                        params,num_periods)

    def _run_single_object(self,object,which_method,params,num_periods):
        neighbor_lightcurves = [(self.objects[self.index_dict[neighbor_ID]].times,
                                     self.objects[self.index_dict[neighbor_ID]].mags,
                                     self.objects[self.index_dict[neighbor_ID]].errs) for neighbor_ID in object.neighbors]



        rv = iterative_deblend(object.times,object.mags,object.errs,
                                   0,3,
                                    function_params=params,
                                    nharmonics_fit=7,
                                    max_fap=.5)


            

if __name__ == "__main__":
    col_a = lc_collection_for_processing(1.,nworkers=1)
    sample_len_1 = 1000
    t1 = np.linspace(0,42,sample_len_1)
    col_a.add_object(t1,10.+np.sin(t1),[.1]*sample_len_1,0.,0.,'object1')
    col_a.add_object(t1,[10.]*(sample_len_1-1) + [10.0001],[.1]*sample_len_1,0.5,0,'object2')


    col_a.run_ls(startp=0.5,endp=4.)
