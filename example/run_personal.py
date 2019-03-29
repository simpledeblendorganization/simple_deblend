'''run_personal.py - Joshua Wallace - Mar 2019

This is an example for how to run the simple_deblend code. It is my personal
example, how I actually run it.

The code takes some command line arguments:

'''

import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc
import glob


if len(sys.argv) != 4:
    print("Expecting n arguments:")
    print(" - path to input light curve files")
    print(" - neighbor inclusion radius")
    print(" - number of control workers")
    exit(-1)




def get_input_light_curves(path_to_input_files):
    '''
    This can be modified as desired to allow for arbitrary input,
    as long as the output format matches what is here.
    '''
    input_files = glob.glob(path_to_input_files + "/*")
    print("Number of input files: " + str(len(input_files)))


    # return format: an iterable collection of (t,lc,err) containers
    return None


def main():
    path_to_input_files = sys.argv[1]
    neighbor_radius = float(sys.argv[2])
    n_control_workers = int(sys.argv[3])


    col = dproc.lc_collection_for_processing(neighbor_radius,
                                             n_control_workers=n_control_workers)

    


    









if __name__ == '__main__':
    main()
