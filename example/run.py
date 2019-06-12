'''run.py - Joshua Wallace - Mar 2019

This is an example for how to run the simple_deblend code.

'''

import numpy as np
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc # one of the simple_deblend codes



def get_input_light_curves(list_of_ids,list_of_times,list_of_mags,list_of_errs):
    '''
    This can be modified as desired to allow for arbitrary input,
    as long as the output format matches what is here.
    '''

    return_dict = {}

    for i in range(len(list_of_ids)):
        return_dict[list_of_ids[i]] = (list_of_times[i],list_of_mags[i],list_of_errs[i])


    # return format: a dictionary of form {ID:(t,lc,err)}
    return return_dict


def get_xy(list_of_ids,list_of_x,list_of_y):
    '''
    This can be modified as desired to allow for arbitrary input,
    as long as the output format matches what is here.
    '''
    
    return_dict = {}

    for i in range(len(list_of_ids)):
        return_dict[list_of_ids[i]] = (list_of_x[i],list_of_y[i])

    # return format: a ditionary of form {ID:(x,y)}
    return return_dict


def sample_threshold(period):
    
    if period < 10.:
        return 2.
    else:
        return 1.


def main():

    # Radius to consider things neighbors
    neighbor_radius = 10

    # Number of master processes
    n_control_workers = 1

    # Minimum and maximum period to search, in days
    min_p = 1.
    max_p = 50.

    # Automatically determine frequency search step size
    autofreq = True

    # These are set to None because autofreq is set to True
    stepsize_ls = None
    stepsize_pdm = None
    stepsize_bls = None
    nphasebins_bls = None

    # Various parameters for the median filtering
    freq_window_epsilon = 4.
    median_filter_window_ls = 80
    median_filter_window_pdm = 80
    median_filter_window_bls = 200

    # Minimum and maximum transit duration for BLS search
    min_transit_duration = 0.01
    max_transit_duration = 0.5


    ##################################################################
    # Now generate sample (fake) light curves

    IDs = ['A','B','C','D']

    n = 1002
    times = [np.linspace(0,90,n),np.linspace(0,90,n),
             np.linspace(0,90,n),np.linspace(0,90,n)]
    mags = [ np.sin(times[0]) + 10., 0.1*np.sin(times[1]) + 10.,
             np.sin(times[2]/2.) + 10., np.sin(times[3]) + 10.]
    errs = [[0.01]*n,[0.01]*n,[0.01]*n,[0.01]*n]
    
    
    x = [0,1,1,100]
    y = [0,0,1,100]


    # Get the light curves
    lcs = get_input_light_curves(IDs,times,mags,errs)

    # Get xy positions
    xy = get_xy(IDs,x,y)


    ##################################################################

    # Initialize the object to be ran
    col = dproc.lc_collection_for_processing(neighbor_radius,
                                             n_control_workers=n_control_workers)

    # Add objects
    for ID in lcs.keys():
        lc = lcs[ID]
        this_xy = xy[ID]
        col.add_object(lc[0],lc[1],lc[2],this_xy[0],this_xy[1],ID)


    
    # Run Lomb-Scargle
    
    col.run_ls(startp=min_p,endp=max_p,autofreq=autofreq,
               stepsize=stepsize_ls,
               sigclip=np.inf,verbose=False,medianfilter=True,
               freq_window_epsilon_mf=freq_window_epsilon,
               freq_window_epsilon_snr=freq_window_epsilon,
               median_filter_size=median_filter_window_ls,
               snr_filter_size=median_filter_window_ls,
               snr_threshold=sample_threshold)


    # Run Phase Dispersion Minimization

    col.run_pdm(startp=min_p,endp=max_p,autofreq=autofreq,
                stepsize=stepsize_pdm,
                sigclip=np.inf,verbose=False,medianfilter=True,
                freq_window_epsilon_mf=freq_window_epsilon,
                freq_window_epsilon_snr=freq_window_epsilon,
                median_filter_size=median_filter_window_pdm,
                snr_filter_size=median_filter_window_pdm,
                snr_threshold=sample_threshold)


    # Run Box-fitting Least Squares

    col.run_bls(startp=min_p,endp=max_p,autofreq=autofreq,
                stepsize=stepsize_bls,
                nphasebins=nphasebins_bls,
                mintransitduration=min_transit_duration,
                maxtransitduration=max_transit_duration,
                sigclip=np.inf,medianfilter=True,
                freq_window_epsilon_mf=freq_window_epsilon,
                freq_window_epsilon_snr=freq_window_epsilon,
                median_filter_size=median_filter_window_bls,
                snr_filter_size=median_filter_window_bls,
                snr_threshold=sample_threshold)
    


if __name__ == '__main__':
    main()
