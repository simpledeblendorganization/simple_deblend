# simple_deblend



[![DOI](https://zenodo.org/badge/192551348.svg)](https://zenodo.org/badge/latestdoi/192551348)

[![Build Status](https://travis-ci.org/simpledeblendorganization/simple_deblend.svg?branch=master)](https://travis-ci.org/simpledeblendorganization/simple_deblend)

This code, originally written by John Hoffman and extended by Joshua
Wallace, is a simple deblending script for performing deblending of
astrophysical light curves.  (Light curves = brightness measurements
over time, e.g. of a star.)  Given a set of light curves,
coordinates for the objects from which the light curves are derived,
and a search radius, variability discovered in any of the light curves
is then compared to similar variability found in nearby objects to see
which objects is the likely true source of the variability.  Any
blended signals found are removed via a multi-harmonic fit and then
the variability search continues.  Ambiguous cases (e.g., more than
one likely source of variability) are handled.

Currently, only Generalized Lomb--Scargle, Phase Dispersion Minimization, and 
Box-fitting Least Squares are implemented for the variability search.

This code is still considered to be in a prototype and development state; 
as such, there may be major bugs as yet present, and the code itself is
still rough in places.  Suggestions, bug reports, comments, etc. are all 
welcome.

## Contents

- **[example](https://github.com/johnh2o2/simple_deblend/tree/master/example)**:
  A directory containing examples of how to run the code.

- **[src](https://github.com/johnh2o2/simple_deblend/tree/master/src)**: 
  A directory containing the source code.

- **[test](https://github.com/johnh2o2/simple_deblend/tree/master/test)**:
  A directory containing tests for the code.

## Dependencies

This code, written in Python, is dependent on the following Python packages: 
NumPy, SciPy, multiprocessing, Matplotlib, and astrobase.  For astrobase in 
particular, v0.4.0 was used in development of the (current) master branch,
and features included in v0.4.1 were used in the development of some of
the other branches.


## License

`simple_deblend` is provided under the MIT license.  See LICENSE file for the 
full text.
