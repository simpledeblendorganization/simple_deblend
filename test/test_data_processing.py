'''test_data_processing.py - Joshua Wallace - Feb 2019

This tests classes and methods in the data_processing.py file.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc
import numpy as np


class test_data_processing_init(unittest.TestCase):

    def setUp(self):
        pass

    def test_lc_collection_setup(self):
        # Test initialization of the lc_collection_for_processing class
        self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                                dproc.lc_collection_for_processing)

    def test_periodsearch_results_setup(self):
        #Test initialization of the periodsearch_results class
        self.assertIsInstance(dproc.periodsearch_results('1'),
                                  dproc.periodsearch_results)

class test_data_processing_run(unittest.TestCase):

    def setUp(self):
        # random number
        rand1 = np.random.RandomState(seed=1844)
        
        ### col_a, very basic, two-object test
        self.col_a = dproc.lc_collection_for_processing(1.,n_control_workers=2)
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


        # another random number
        rand2 = np.random.RandomState(seed=1847)
        
        ### col_b, more full test with some blending
        self.col_b = dproc.lc_collection_for_processing(5.,n_control_workers=1)#2)
        sample_len_2 = 5000
        t2 = np.linspace(0,1000,sample_len_2)
        self.omega1 = 1.
        phi1 = 0.
        self.omega2 = .582927
        self.omega3 = 1.2023432
        phi2 = .3422982
        sigma1 = .01
        self.col_b.add_object(t2,3. + np.sin(self.omega1*t2+phi1) +
                                  .02*np.sin(self.omega2*t2+phi2) +
                                  sigma1*rand2.randn(sample_len_2),
                                  [sigma1]*sample_len_2,
                                  0.,0.,'o1')
        sigma2 = .03
        self.col_b.add_object(t2,3. + .12*np.sin(self.omega3*t2+phi1) +
                                  .5*np.sin(self.omega2*t2+phi2) +
                                  sigma2*rand2.randn(sample_len_2),
                                  [sigma2]*sample_len_2,
                                  0.,0.0001,'o2')
        sigma3 = .05
        self.col_b.add_object(t2,3.+.94*np.sin(self.omega2*t2+phi2) +
                                  sigma3*rand2.randn(sample_len_2),
                                  [sigma3]*sample_len_2,3.,3.99,'o3')
        sigma4 = .01
        self.col_b.add_object(t2,[3.5]*sample_len_2,
                                  [sigma4]*sample_len_2,2.,1.5,'o4')
        sigma5 = .07
        self.col_b.add_object(t2,2. + 3.*np.sin(self.omega2*t2+phi2) +
                                  .5*np.sin(self.omega1*t2+phi1) +
                                  sigma5*rand2.randn(sample_len_2),
                                  [sigma5]*sample_len_2,10000,10000,'o5')

        ### col_c, test with some blending
        rand3 = np.random.RandomState(seed=1847)
        self.col_c = dproc.lc_collection_for_processing(5,n_control_workers=2)
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

                             
        
    def test_basic_ls_run(self):
        # Test a basic run of the iterative deblending
        self.col_a.run_ls(startp=6.,endp=7.,stepsize=0.0000001,autofreq=False,max_fap=.4)

        with self.assertRaises(KeyError):
            self.col_a.results['object1']['BLS']
        with self.assertRaises(KeyError):
            self.col_a.results['object2']['LS']
        self.assertEqual(len(self.col_a.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a.results['object1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi,places=6)
        self.assertEqual(len(self.col_a.results['object1']),1)
        self.assertEqual(len(self.col_a.results['object1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_a.results['object1']['LS'].blends_info),0)
        self.assertEqual(len(self.col_a.results['object2']),0)

        
    def test_simple_blended_ls_run(self):
        self.col_c.run_ls(startp=0.5,endp=2.,stepsize=5e-5,autofreq=False,max_fap=.1)

        # Check c1
        self.assertEqual(len(self.col_c.results['c1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c1']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c1']['LS'].good_periods_info[0]['num_previous_blends'],0)

        # Check c2
        self.assertEqual(len(self.col_c.results['c2']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_c.results['c2']['LS'].blends_info),1)
        self.assertEqual(self.col_c.results['c2']['LS'].blends_info[0]['ID_of_blend'],'c1')
        self.assertAlmostEqual(self.col_c.results['c2']['LS'].blends_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)

        #Check c3
        self.assertEqual(len(self.col_c.results['c3']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c3']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c3']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c3']['LS'].good_periods_info[0]['num_previous_blends'],0)


    def test_longperiod(self):
        # Test a long period object
        self.col_d.run_ls(startp=28.,endp=32.,autofreq=True)

        self.assertEqual(len(self.col_d.results['d1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_d.results['d1']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_d.results['d1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegad,places=5)

    def test_multipleblend(self):
        # Test a long period object and also objects with multiple blends
        self.col_e.run_ls(startp=2.5,endp=4.7,max_fap=.23,autofreq=True)#,stepsize=1e-7)

        self.assertEqual(len(self.col_e.results['e1']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_e.results['e1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegae,places=3)
        self.assertEqual(len(self.col_e.results['e2']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e3']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e4']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e2']['LS'].blends_info),1)
        self.assertEqual(len(self.col_e.results['e3']['LS'].blends_info),1)
        self.assertEqual(len(self.col_e.results['e4']['LS'].blends_info),1)
        self.assertEqual(self.col_e.results['e2']['LS'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e3']['LS'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e4']['LS'].blends_info[0]['ID_of_blend'],'e1')




        
    def _test_blended_ls_run(self):
        self.col_b.run_ls(startp=2.,endp=11.,stepsize=1e-5,autofreq=False)

        # Check o1
        self.assertEqual(len(self.col_b.results['o1']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_b.results['o1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi*self.omega1,places=3)
        self.assertEqual(len(self.col_b.results['o1']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_b.results['o1']['LS'].blends_info[0]['lsp_dict']['bestperiod'],2*np.pi*self.omega2,places=3)
        self.assertEqual(self.col_b.results['o1']['LS'].blends_info[0]['ID_of_blend'],'o3')

        # Check o2
        self.assertEqual(len(self.col_b.results['o2']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_b.results['o2']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi*self.omega3,places=3)
        self.assertEqual(len(self.col_b.results['o2']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_b.results['o2']['LS'].blends_info[0]['lsp_dict']['bestperiod'],2*np.pi*self.omega2,places=3)
        self.assertEqual(self.col_b.results['o2']['LS'].blends_info[0]['ID_of_blend'],'o3')

        # Check o3
        self.assertEqual(len(self.col_b.results['o3']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_b.results['o3']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_b.results['o3']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2*np.pi.self.omega2,places=3)
        
        # Check o4
        with self.assertRaises(KeyError):
            self.col_b.results['o4']['LS']
        self.assertEqual(len(self.col_b.results['o4'].keys()),0)


if __name__ == '__main__':
    unittest.main()

