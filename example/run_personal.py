'''run_personal.py - Joshua Wallace - Mar 2019

This is an example for how to run the simple_deblend code. It is my personal
example, how I actually run it.

The code takes some command line arguments:

'''

import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc
import glob
import pickle
from astrobase.lcmath import sigclip_magseries
import numpy as np
import personal_thresholds as thresh


print(sys.argv)


if len(sys.argv) != 16 and not (len(sys.argv) == 17 and sys.argv[-1] == '\\'):
    print("Expecting 14 arguments:")
    print(" - path to input light curve files")
    print(" - neighbor inclusion radius")
    print(" - number of control workers")
    print(" - file with x,y position information")
    print(" - starting period")
    print(" - ending period")
    print(" - autofreq")
    print(" - stepsize LS")
    print(" - stepsize PDM")
    print(" - stepsize BLS")
    print(" - freq_window_epsilon")
    print(" - median filter window LS and PDM")
    print(" - median filter window BLS")
    print(" - min transit duration")
    print(" - max transit duration")
    exit(-1)




def get_input_light_curves(path_to_input_files):
    '''
    This can be modified as desired to allow for arbitrary input,
    as long as the output format matches what is here.
    '''
    input_files = glob.glob(path_to_input_files + "/*")
    input_files = input_files[:10]
    print("Number of input files: " + str(len(input_files)))

    return_dict = {}
    number_skipped = 0

    for f in input_files:
        t, lc, err = pickle.load(open(f,"rb"),encoding='latin')
        if len(t) < 800:
            number_skipped += 1
            continue
        t, lc, err = sigclip_magseries(t,lc,err,sigclip=3,iterative=True,
                                 niterations=3)
        return_dict[f.split("/")[-1].split("_")[0]] = \
            (t,lc,err)
            
    print("Number skipped for too few points: " + str(number_skipped))


    # return format: a dictionary of form ID:(t,lc,err) 
    return return_dict


def get_xy(file_xy):
    '''
    This can be modified as desired to allow for arbitrary input,
    as long as the output format matches what is here.
    '''
    
    with open(file_xy,'r') as f:
        lines = f.readlines()

    return_dict = {}
    for line in lines:
        s = line.split()
        if float(s[3]) < 19.:
            return_dict[s[0]] = (float(s[-2]),float(s[-1]))

    # return format: a ditionary of form ID:(x,y)
    return return_dict


def main():
    path_to_input_files = sys.argv[1]
    neighbor_radius = float(sys.argv[2])
    n_control_workers = 1#int(sys.argv[3])
    file_xy = sys.argv[4]
    start_p = float(sys.argv[5])
    end_p = float(sys.argv[6])
    if sys.argv[7] in ['true','True','t','T']:
        autofreq = True
    elif sys.argv[7] in ['false','False','f','F']:
        autofreq = False
    else:
        raise ValueError("did not recognize autofreq value")
    if autofreq:
        if sys.argv[8] not in ['None','none','-']:
            raise ValueError("autofreq is True, but stepsize_ls was not None")
        if sys.argv[9] not in ['None','none','-']:
            raise ValueError("autofreq is True, but stepsize_pdm was not None")
        if sys.argv[10] not in ['None','none','-']:
            raise ValueError("autofreq is True, but stepsize_bls was not None")
        stepsize_ls = None#np.inf
        stepsize_pdm = None#np.inf
        stepsize_bls = None#np.inf
    else:
        stepsize_ls = float(sys.argv[8])
        stepsize_pdm = float(sys.argv[9])
        stepsize_bls = float(sys.argv[10])
    freq_window_epsilon = float(sys.argv[11])
    median_filter_window_ls = int(sys.argv[12])
    median_filter_window_pdm = median_filter_window_ls
    median_filter_window_bls = int(sys.argv[13])
    min_transit_duration = float(sys.argv[14])
    max_transit_duration = float(sys.argv[15])

    # Get the light curves
    lcs = get_input_light_curves(path_to_input_files)

    # Get xy positions
    xy = get_xy(file_xy)

    # Initialize the object to be ran
    col = dproc.lc_collection_for_processing(neighbor_radius,
                                             n_control_workers=n_control_workers)
    # Add objects
    for ID in lcs.keys():
        lc = lcs[ID]
        this_xy = xy[ID]
        col.add_object(lc[0],lc[1],lc[2],this_xy[0],this_xy[1],ID)


    
    # Run Lomb-Scargle
    
    col.run_ls(startp=start_p,endp=end_p,autofreq=autofreq,
               stepsize=stepsize_ls,
               sigclip=np.inf,verbose=False,medianfilter=True,
               freq_window_epsilon_mf=freq_window_epsilon,
               freq_window_epsilon_snr=freq_window_epsilon,
               median_filter_size=median_filter_window_ls,
               snr_filter_size=median_filter_window_ls,
               snr_threshold=thresh.ls_cutoff)

    col.run_pdm(startp=start_p,endp=end_p,autofreq=autofreq,
                stepsize=stepsize_pdm,
                sigclip=np.inf,verbose=False,medianfilter=True,
                freq_window_epsilon_mf=freq_window_epsilon,
                freq_window_epsilon_snr=freq_window_epsilon,
                median_filter_size=median_filter_window_pdm,
                snr_filter_size=median_filter_window_pdm,
                snr_threshold=thresh.pdm_cutoff)

    
    col.run_bls(startp=start_p,endp=end_p,autofreq=autofreq,
                stepsize=stepsize_bls,
                nphasebins=200,
                mintransitduration=min_transit_duration,
                maxtransitduration=max_transit_duration,
                sigclip=np.inf,medianfilter=True,
                freq_window_epsilon_mf=freq_window_epsilon,
                freq_window_epsilon_snr=freq_window_epsilon,
                median_filter_size=median_filter_window_bls,
                snr_filter_size=median_filter_window_bls,
                snr_threshold=thresh.bls_cutoff)
    
    


    col.save_periodsearch_results(".")



"""
6045466193420224896
  PDM PERIOD: 1.71874e+00 days;  pSNR: 2.10418e+01
    checking blends
    6045466193420210816
     n: 1.0537190427208654e-07 vs.  1.0323214044661785e-20
   -> blended! Trying again.
6045466193420225536
  PDM PERIOD: 6.64019e-02 days;  pSNR: 7.04094e+01
    checking blends
    6045466193420214016
     n: 5.093358098759847e-07 vs.  2.0844951436336297e-20
   -> blended! Trying again.
6045466193420224896
  PDM PERIOD: 1.71874e+00 days;  pSNR: 2.10418e+01
    checking blends
    6045466193420210816
     n: 1.0537190427208654e-07 vs.  1.376428539288238e-20
   -> blended! Trying again.
6045466193420225536
  PDM PERIOD: 6.64019e-02 days;  pSNR: 7.04094e+01
    checking blends
    6045466193420214016
     n: 5.093358098759847e-07 vs.  1.1432297809972885e-19
   -> blended! Trying again.
6045466193420224896
  PDM PERIOD: 1.71874e+00 days;  pSNR: 2.10418e+01
    checking blends
    6045466193420210816
     n: 1.0537190427208654e-07 vs.  8.946785505373547e-21
   -> blended! Trying again.
6045466193420225536
  PDM PERIOD: 6.64019e-02 days;  pSNR: 7.04094e+01
    checking blends
    6045466193420214016
     n: 5.093358098759847e-07 vs.  2.1146706923795642e-19
   -> blended! Trying again.
6045466193420224896
  PDM PERIOD: 1.71874e+00 days;  pSNR: 2.10418e+01
    checking blends
    6045466193420210816
     n: 1.0537190427208654e-07 vs.  1.1011428314305904e-20
   -> blended! Trying again.
6045466193420225536
  PDM PERIOD: 6.64019e-02 days;  pSNR: 7.04094e+01
    checking blends
    6045466193420214016
     n: 5.093358098759847e-07 vs.  1.3998542942357398e-19
   -> blended! Trying again.
6045466193420224896
  PDM PERIOD: 1.71874e+00 days;  pSNR: 2.10418e+01
    checking blends
    6045466193420210816
     n: 1.0537190427208654e-07 vs.  8.946785505373547e-21
   -> blended! Trying again.
   Reached the blend recursion level, no longer checking
6045466193420225536
  PDM PERIOD: 6.64019e-02 days;  pSNR: 7.04094e+01
    checking blends
    6045466193420214016
     n: 5.093358098759847e-07 vs.  4.765883817285524e-20
   -> blended! Trying again.
   Reached the blend recursion level, no longer checking
"""






if __name__ == '__main__':
    main()
