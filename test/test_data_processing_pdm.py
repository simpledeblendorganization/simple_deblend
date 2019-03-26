'''test_data_processing.py - Joshua Wallace - Feb 2019

This tests classes and methods in the data_processing.py file,
using the phase dispersion minimization method.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc
import numpy as np


class test_data_processing_pdm_sinusoidal_single_signal(unittest.TestCase):

    def setUp(self):
        # random number
        rand1 = np.random.RandomState(seed=1844)
        
        ### col_a, very basic, two-object test
        self.col_a = dproc.lc_collection_for_processing(1.,n_control_workers=1)#2)
        sample_len_1 = 3000
        sigma1 = .01
        t1 = np.linspace(0,1200,sample_len_1)
        self.col_a.add_object(t1,10.+np.sin(t1)+sigma1*rand1.randn(sample_len_1),
                                  [sigma1]*sample_len_1,0.,0.,'object1')
        # Modify one value 
        #self.col_a.objects[0].mags[3] =\
        #                self.col_a.objects[0].mags[3] + .001
        self.col_a.add_object(t1,np.array([10.]*sample_len_1)+sigma1*rand1.randn(sample_len_1),
                                  [sigma1]*sample_len_1,0.5,0,'object2')



        ### col_c, test with some blending
        rand3 = np.random.RandomState(seed=1847)
        self.col_c = dproc.lc_collection_for_processing(5,n_control_workers=1)#2)
        sample_len_3=3500
        t3 = np.linspace(0,80,sample_len_3)
        self.omegac = 8.319
        phi_c = .2341
        sigmac1 = .05
        sigmac2 = .07
        sigmac3 = .2
        temp1 = rand3.randn(sample_len_3)
        temp2 = rand3.randn(sample_len_3)
        _ = rand3.randn(sample_len_3)
        temp3 = rand3.randn(sample_len_3)
        self.col_c.add_object(t3,8. + np.sin(self.omegac*t3 + phi_c) +
                                  sigmac1*temp1,
                                  [sigmac1]*sample_len_3,0.,1.,'c1')
        self.col_c.add_object(t3, 10. + 0.5*np.sin(self.omegac*t3 + phi_c) +
                                  sigmac2*temp2,
                                  [sigmac2]*sample_len_3,1.,2.5,'c2')
        self.col_c.add_object(t3,4. + 0.1*np.sin(self.omegac*t3 + phi_c) +
                                  sigmac3*temp3,
                                  [sigmac3]*sample_len_3,10234.,-2327912.7,'c3')


        ### col_d, long-period test 
        rand4 = np.random.RandomState(seed=1853)
        self.col_d = dproc.lc_collection_for_processing(10,n_control_workers=1)
        
        sample_len_4 = 5002
        t4 = np.linspace(0,102,sample_len_4)
        self.omegad = 2.*np.pi/30.1107
        sigma4 = .2
        self.col_d.add_object(t4, 5.+.75*sigma4*np.sin(self.omegad*t4)+sigma4*rand4.randn(sample_len_4),
                                  [sigma4]*sample_len_4,0.,1.0,'d1')

        ### col_e, multiple blended objects
        rand5 = np.random.RandomState(seed=1893)
        self.col_e = dproc.lc_collection_for_processing(10.,n_control_workers=1)

        sample_len_5 = 2855
        t5 = np.linspace(0,21.5,sample_len_5)
        self.omegae = 2.*np.pi/3.48208#3.47464
        phi_e = .3
        sigma5 = .1
        self.col_e.add_object(t5, 4.+np.sin(self.omegae*t5+phi_e)+sigma5*rand4.randn(sample_len_5),
                                  [sigma5]*sample_len_5,0.,0.5,'e1')
        self.col_e.add_object(t5, 4.+.4*np.sin(self.omegae*t5+phi_e)+sigma5*rand4.randn(sample_len_5),
                                  [sigma5]*sample_len_5,0.5,0.,'e2')
        self.col_e.add_object(t5, 4.+.3*np.sin(self.omegae*t5+phi_e)+sigma5*rand4.randn(sample_len_5),
                                  [sigma5]*sample_len_5,0.7,0.1,'e3')
        self.col_e.add_object(t5, 4.+.2*np.sin(self.omegae*t5+phi_e)+sigma5*rand4.randn(sample_len_5),
                                  [sigma5]*sample_len_5,3.0,0.9,'e4')

    def test_lc_collection_setup(self):
        # Test initialization of the lc_collection_for_processing class
        self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                                dproc.lc_collection_for_processing)

    def test_periodsearch_results_setup(self):
        #Test initialization of the periodsearch_results class
        self.assertIsInstance(dproc.periodsearch_results('1'),
                                  dproc.periodsearch_results)

                             
    ### To be added back in when signal vetting is good
    def test_basic_pdm_run(self):
        # Test a basic run of the iterative deblending
        self.col_a.run_pdm(startp=6.,endp=7.,stepsize=0.0000001,
                           autofreq=False,max_fap=.4,medianfilter=False,nworkers=1)

        with self.assertRaises(KeyError):
            self.col_a.results['object1']['BLS']
        #with self.assertRaises(KeyError):  #### To be added back in later when signal vetting is working
        #    self.col_a.results['object2']['PDM']
        #self.assertEqual(len(self.col_a.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a.results['object1']['PDM'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi,places=4)
        self.assertEqual(len(self.col_a.results['object1']),1)
        self.assertEqual(len(self.col_a.results['object1']['PDM'].good_periods_info),1)
        self.assertEqual(len(self.col_a.results['object1']['PDM'].blends_info),0)
        #self.assertEqual(len(self.col_a.results['object2']),0)

    #### To be added back in when signal vetting is good
    def __test_simple_blended_pdm_run(self):
        self.col_c.run_pdm(startp=0.5,endp=2.,stepsize=5e-5,
                           autofreq=False,max_fap=.1,medianfilter=False)

        # Check c1
        self.assertEqual(len(self.col_c.results['c1']['PDM'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c1']['PDM'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c1']['PDM'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c1']['PDM'].good_periods_info[0]['num_previous_blends'],0)

        # Check c2
        self.assertEqual(len(self.col_c.results['c2']['PDM'].good_periods_info),0)
        self.assertEqual(len(self.col_c.results['c2']['PDM'].blends_info),1)
        self.assertEqual(self.col_c.results['c2']['PDM'].blends_info[0]['ID_of_blend'],'c1')
        self.assertAlmostEqual(self.col_c.results['c2']['PDM'].blends_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)

        #Check c3
        self.assertEqual(len(self.col_c.results['c3']['PDM'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c3']['PDM'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c3']['PDM'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c3']['PDM'].good_periods_info[0]['num_previous_blends'],0)


    def __test_longperiod(self):
        # Test a long period object
        self.col_d.run_pdm(startp=28.,endp=32.,autofreq=True,
                           medianfilter=False)

        self.assertEqual(len(self.col_d.results['d1']['PDM'].good_periods_info),1)
        self.assertEqual(len(self.col_d.results['d1']['PDM'].blends_info),0)
        self.assertAlmostEqual(self.col_d.results['d1']['PDM'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegad,places=5)

    def __test_multipleblend(self):
        # Test a long period object and also objects with multiple blends
        self.col_e.run_pdm(startp=2.5,endp=4.7,max_fap=.23,autofreq=True,
                           medianfilter=False)

        self.assertEqual(len(self.col_e.results['e1']['PDM'].good_periods_info),1)
        self.assertAlmostEqual(self.col_e.results['e1']['PDM'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegae,places=3)
        self.assertEqual(len(self.col_e.results['e2']['PDM'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e3']['PDM'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e4']['PDM'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e2']['PDM'].blends_info),1)
        self.assertEqual(len(self.col_e.results['e3']['PDM'].blends_info),1)
        self.assertEqual(len(self.col_e.results['e4']['PDM'].blends_info),1)
        self.assertEqual(self.col_e.results['e2']['PDM'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e3']['PDM'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e4']['PDM'].blends_info[0]['ID_of_blend'],'e1')



class test_data_processing_pdm_rrlyrae_signal(unittest.TestCase):

    def setUp(self):

        self.rr_period = 0.490751
        self.fourier_amp = np.array([-0.09340915, -0.28991142, -0.14849184,
                                     -0.1043798 , -0.06841426,-0.04780541, 
                                     -0.02871969, -0.01738949, -0.01293884,  
                                     0.00956898,   0.00611337,  0.00324416])
        self.fourier_phase=np.array([ 0.12177694, -0.56953169, -0.34281352,  
                                      0.21209113,  0.70671323, 1.31876177,  
                                      1.89734647,  2.28429027,  2.66283256,  
                                      0.10880013,  0.75884711,  1.30690127])
        time_length = 3000
        t = np.linspace(0,75.21,time_length)
        phase = t/self.rr_period
        phase = phase - np.floor(phase)
        fseries = np.array([self.fourier_amp[i]*np.cos(2.0*np.pi*i*phase + 
                                              self.fourier_phase[i])
                   for i in range(len(self.fourier_amp))])

        # Then add this to the mags
        rand = np.random.RandomState(seed=1820)
        sigma1 = .01
        mags_1 = 13.5 + sigma1*rand.randn(time_length)
        for fo in fseries:
            mags_1 += fo

        sigma2 = .02
        mags_2 = 16. + sigma2*rand.randn(time_length)
        for fo in fseries:
            mags_2 += .1*fo

        self.col = dproc.lc_collection_for_processing(5.,n_control_workers=1)
        self.col.add_object(t,mags_1,[sigma1]*time_length,0.,0.,'rr1')
        self.col.add_object(t,mags_2,[sigma2]*time_length,1.,1.6,'rr2')


    def __test_rrblend(self):
        # Test an RRab and a blend
        self.col.run_pdm(startp=.2,endp=1.,stepsize=1e-4,autofreq=False,
                         medianfilter=False)

        self.assertEqual(len(self.col.results['rr1']['PDM'].good_periods_info),1)
        self.assertEqual(len(self.col.results['rr1']['PDM'].blends_info),0)
        self.assertAlmostEqual(self.col.results['rr1']['PDM'].good_periods_info[0]['lsp_dict']['bestperiod'],self.rr_period,places=2)
        self.assertEqual(len(self.col.results['rr2']['PDM'].good_periods_info),0)
        self.assertEqual(len(self.col.results['rr2']['PDM'].blends_info),1)
        self.assertEqual(self.col.results['rr2']['PDM'].blends_info[0]['ID_of_blend'],'rr1')
        

if __name__ == '__main__':
    unittest.main()
