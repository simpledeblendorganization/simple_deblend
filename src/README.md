This directory contains the source code for this repository.

 - `data_processing.py`: the main wrapper functions to interface between user
   and the code in `simple_deblend.py`.

 - `light_curve_class.py`: code that defines the main data storage classes.

 - `pinknoise.py`: code used to calculate signal-to-pink-noise for BLS
   results.  Still under development.

 - `simple_deblend.py`: the main code that runs the period search,
    blend detection, and signal vetting.

 - `snr_calculation.py`: some code to run median filtering of the periodogram
    and calculating the periodogram signal to noise.