'''test_simple_deblend_functions.py - Joshua Wallace - Mar 2019

This tests functions in simple_deblend.py
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import simple_deblend as sdb
import numpy as np


class test_median_filter(unittest.TestCase):

    def setUp(self):
        n = 5000
        self.val = 2.
        self.lspvals = self.val + np.zeros(n)
        frequencies = np.linspace(.05,10,n)
        self.periods = 1./frequencies

    def test_filter_doesnt_mess_up_flat(self):
        output = sdb.median_filtering(self.lspvals,self.periods,
                                      2.,30,60.)
        for val1, val2 in zip(output,self.lspvals-self.val):
            self.assertAlmostEqual(val1,val2,places=10)

        mod_lspvals = self.lspvals[:]
        high_val = 20.
        mod_lspvals[len(mod_lspvals)//2] = high_val
        output2 = sdb.median_filtering(mod_lspvals,self.periods,2.,30,60.)
        for val1, val2 in zip(output2,mod_lspvals-self.val):
            self.assertAlmostEqual(val1,val2,places=11)

    def test_filter_cleans_up_slope(self):
        addon_min = 0.
        addon_max = 1000.
        addon_max = addon_max - (addon_max-addon_min)/len(self.lspvals)
        lspvals = self.lspvals +\
                  np.linspace(addon_min,addon_max,len(self.lspvals))
        median_filter_size = 4
        output_slope = sdb.median_filtering(lspvals,self.periods,1.,
                                            median_filter_size,502.)
        for i in range(len(output_slope)):
            if i < median_filter_size:
                self.assertLess(output_slope[i],0.)
            elif len(output_slope)-i < median_filter_size+1:
                self.assertGreater(output_slope[i],0.)
            else:
                self.assertAlmostEqual(output_slope[i],0.,places=8)


if __name__ == '__main__':
    unittest.main()
