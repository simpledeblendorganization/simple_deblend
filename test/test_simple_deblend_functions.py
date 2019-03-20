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
        self.assertAlmostEqual(output,self.lspvals-self.val)
