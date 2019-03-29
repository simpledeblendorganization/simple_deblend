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


if len(sys.argv) != 5:
    print("Expecting 4 arguments:")
    print(" - path to input light curve files")
    print(" - neighbor inclusion radius")
    print(" - number of control workers")
    print(" - file with x,y position information")
    exit(-1)




def get_input_light_curves(path_to_input_files):
    '''
    This can be modified as desired to allow for arbitrary input,
    as long as the output format matches what is here.
    '''
    input_files = glob.glob(path_to_input_files + "/*")
    print("Number of input files: " + str(len(input_files)))

    return_dict = {}
    number_skipped = 0

    for f in input_files:
        t, lc, err = pickle.load(open(f,"rb"),encoding='latin')
        if len(t) < 800:
            number_skipped += 1
            continue
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
        print(s[0])
        if float(s[3]) < 19.:
            return_dict[s[0]] = (float(s[-2]),float(s[-1]))

    # return format: a ditionary of form ID:(x,y)
    return return_dict


def main():
    path_to_input_files = sys.argv[1]
    neighbor_radius = float(sys.argv[2])
    n_control_workers = int(sys.argv[3])
    file_xy = sys.argv[4]

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

    


    









if __name__ == '__main__':
    main()
