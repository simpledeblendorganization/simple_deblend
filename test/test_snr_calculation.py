'''test_snr_calculation.py - Joshua Wallace - Mar 2019

This tests the SNR calculation.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import snr_calculation as snr
import numpy as np


class test_periodogram_snr_setup(unittest.TestCase):

    def setUp(self):
        pass

    def test_errors(self):
        with self.assertRaises(ValueError):
            snr.periodogram_snr([1.,2.,3.,4.,5.],[3.,4.,5.],1,1,'LS')
        with self.assertRaises(AttributeError):
            snr.periodogram_snr([1,2],[1,2],[1,2],23,'LS')
        with self.assertRaises(ValueError):
            snr.periodogram_snr([1,np.inf],[1,2],1,4,'PDM')
        with self.assertRaises(ValueError):
            snr.periodogram_snr([1,np.nan],[1,2],1,5,'PDM')
        with self.assertRaises(AttributeError):
            snr.periodogram_snr([1,2],[1,2],[1,2],2,'bob__')



class test_ls_snr(unittest.TestCase):
    
    def setUp(self):
        rand1 = np.random.RandomState(seed=47)

        basevalue = .5
        self.per_delta = 0.1
        self.per_pdm_delta = 0.03
        self.per_amplitude = 10.
        self.pdm_per_amplitude = .1
        per_length = 10000
        per_min = 1.
        per_max = 10.

        self.periodogram = np.full(per_length,basevalue)
        for i in range(len(self.periodogram)):
            if i % 2 == 1:
                self.periodogram[i] = basevalue + self.per_delta
        self.index = per_length//2
        self.periodogram[self.index] = self.per_amplitude
        self.periods = np.linspace(1./per_max,1./per_min,per_length)
        self.periods = 1./self.periods


        self.pdm_periodogram = np.full(per_length,1.)
        for i in range(len(self.periodogram)):
            if i % 2 == 1:
                self.pdm_periodogram[i] = 1. - self.per_pdm_delta
        self.pdm_periodogram[self.index] = self.pdm_per_amplitude

    def test_ls_snr(self):
        snr_val = snr.periodogram_snr(self.periodogram,self.periods,self.index,
                                      80.,'LS')
        self.assertAlmostEqual(snr_val,self.per_amplitude/(self.per_delta/2.),places=8)

    def test_bls_snr(self):
        snr_val = snr.periodogram_snr(self.periodogram,self.periods,self.index,
                                      80.,'BLS')
        self.assertAlmostEqual(snr_val,self.per_amplitude/(self.per_delta/2.),places=8)

    def test_pdm_snr(self):
        snr_val = snr.periodogram_snr(self.pdm_periodogram,self.periods,
                                      self.index,80.,'PDM')
        self.assertAlmostEqual(snr_val,(1. - self.pdm_per_amplitude)/(self.per_pdm_delta/2.),places=8)



if __name__ == '__main__':
    unittest.main()
