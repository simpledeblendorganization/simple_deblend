# simple_deblend

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

Currently, only Lomb--Scargle is implemented for the variability
search.  Phase dispersion minimization and box least squares are planned.
