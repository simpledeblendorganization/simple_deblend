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
        self.val_pdm = .9
        self.lspvals_pdm = self.val_pdm + np.zeros(n)
        frequencies = np.linspace(.05,10,n)
        self.periods = 1./frequencies

    def test_filter_doesnt_mess_up_flat(self):
        output = sdb.median_filtering(self.lspvals,self.periods,
                                      2.,30,60.,'LS')
        for val1, val2 in zip(output,self.lspvals-self.val):
            self.assertAlmostEqual(val1,val2,places=10)

        mod_lspvals = self.lspvals[:]
        high_val = 20.
        mod_lspvals[len(mod_lspvals)//2] = high_val
        output2 = sdb.median_filtering(mod_lspvals,self.periods,2.,30,60.,'LS')
        for val1, val2 in zip(output2,mod_lspvals-self.val):
            self.assertAlmostEqual(val1,val2,places=11)

    def test_filter_doesnt_mess_up_flat_pdm(self):
        output = sdb.median_filtering(self.lspvals_pdm,self.periods,
                                      2.,30,60.,'PDM')
        for val1, val2 in zip(output,self.lspvals_pdm - self.val_pdm + 1.):
            self.assertAlmostEqual(val1,val2,places=11)

        mod_lspvals_pdm = self.lspvals_pdm[:]
        high_val = .1
        mod_lspvals_pdm[len(mod_lspvals_pdm)//2] = high_val
        output2 = sdb.median_filtering(mod_lspvals_pdm,self.periods,
                                       2.,30,60.,'PDM')
        for val1, val2 in zip(output2,mod_lspvals_pdm - self.val_pdm + 1.):
            self.assertAlmostEqual(val1,val2,places=11)

    def test_filter_cleans_up_slope(self):
        addon_min = 0.
        addon_max = 1000.
        addon_max = addon_max - (addon_max-addon_min)/len(self.lspvals)
        lspvals = self.lspvals +\
                  np.linspace(addon_min,addon_max,len(self.lspvals))
        median_filter_size = 4
        output_slope = sdb.median_filtering(lspvals,self.periods,1.,
                                            median_filter_size,502.,'LS')
        for i in range(len(output_slope)):
            if i < median_filter_size:
                self.assertLess(output_slope[i],0.)
            elif len(output_slope)-i < median_filter_size+1:
                self.assertGreater(output_slope[i],0.)
            else:
                self.assertAlmostEqual(output_slope[i],0.,places=8)



class test_flux_amplitude(unittest.TestCase):

    def setUp(self):
        self.n = 5000
        self.freq = 1./4.
        self.t1 = np.linspace(0,100,self.n)
        self.m1_middle = 12.5
        self.m2_middle = 17.5
        self.m3_middle = 12.6
        self.m4_middle = 18.0
        self.m5_middle = self.m1_middle+5.
        self.sigma = 0.005

        rand1 = np.random.RandomState(seed=78)
        self.m1 = self.m1_middle + 0.5*np.sin(2.*np.pi*self.freq*self.t1) + self.sigma*rand1.randn(self.n)

        rand2 = np.random.RandomState(seed=79)
        self.m2 = self.m2_middle + 0.5*np.sin(2.*np.pi*self.freq*self.t1) + self.sigma*rand2.randn(self.n)
        rand3 = np.random.RandomState(seed=80)
        self.m3 = self.m3_middle + 0.5*np.sin(2.*np.pi*self.freq*self.t1) + self.sigma*rand3.randn(self.n)
        rand4 = np.random.RandomState(seed=81)
        self.m4 = self.m4_middle + 2.*np.sin(2.*np.pi*self.freq*self.t1) + self.sigma*rand4.randn(self.n)

        rand5 = np.random.RandomState(seed=82)
        self.m5 = self.m5_middle + 2.5*np.sin(2.*np.pi*self.freq*self.t1) + self.sigma*rand5.randn(self.n)

    def test_(self):

        ff1 = sdb.FourierFit(nharmonics=3).fit(self.t1,self.m1,np.array([self.sigma]*self.n),self.freq)
        ff2 = sdb.FourierFit(nharmonics=3).fit(self.t1,self.m2,np.array([self.sigma]*self.n),self.freq)
        ff3 = sdb.FourierFit(nharmonics=3).fit(self.t1,self.m3,np.array([self.sigma]*self.n),self.freq)
        ff4 = sdb.FourierFit(nharmonics=3).fit(self.t1,self.m4,np.array([self.sigma]*self.n),self.freq)
        ff5 = sdb.FourierFit(nharmonics=3).fit(self.t1,self.m5,np.array([self.sigma]*self.n),self.freq)

        self.assertEqual(ff1.flux_amplitude(),ff1.flux_amplitude(return_df_f0=True)[0])
        self.assertEqual(ff2.flux_amplitude(),ff2.flux_amplitude(return_df_f0=True)[0])
        self.assertEqual(ff3.flux_amplitude(),ff3.flux_amplitude(return_df_f0=True)[0])
        self.assertEqual(ff4.flux_amplitude(),ff4.flux_amplitude(return_df_f0=True)[0])
        self.assertEqual(ff5.flux_amplitude(),ff5.flux_amplitude(return_df_f0=True)[0])

        self.assertGreater(ff1.flux_amplitude(),ff2.flux_amplitude())
        self.assertGreater(ff1.flux_amplitude(),ff3.flux_amplitude())
        self.assertGreater(ff1.flux_amplitude(),ff4.flux_amplitude())
        self.assertGreater(ff4.flux_amplitude(),ff2.flux_amplitude())

        self.assertAlmostEqual(ff1.flux_amplitude(return_df_f0=True)[1],ff2.flux_amplitude(return_df_f0=True)[1],places=2)
        self.assertAlmostEqual(ff1.flux_amplitude(return_df_f0=True)[1],ff3.flux_amplitude(return_df_f0=True)[1],places=2)

        # See if we can get reasonably close, as we would expect based on the math
        self.assertLess(abs(ff1.flux_amplitude()-10.*ff5.flux_amplitude())/ff1.flux_amplitude(), .1)
                               
        self.assertGreater(ff5.flux_amplitude(return_df_f0=True)[1],ff1.flux_amplitude(return_df_f0=True)[1])


if __name__ == '__main__':
    unittest.main()
