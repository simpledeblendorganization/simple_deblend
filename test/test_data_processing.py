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
        ### col_a, very basic, two-object test
        self.col_a = dproc.lc_collection_for_processing(1.,n_control_workers=2)
        sample_len_1 = 3000
        t1 = np.linspace(0,1200,sample_len_1)
        self.col_a.add_object(t1,10.+np.sin(t1),[.1]*sample_len_1,0.,0.,'object1')
        # Modify one value 
        self.col_a.objects[0].mags[3] =\
                        self.col_a.objects[0].mags[3] + .001
        self.col_a.add_object(t1,[10.]*(sample_len_1-1) + [10.0001],[.1]*sample_len_1,0.5,0,'object2')

        ### col_b, more full test with some blending
        self.col_b = dproc.lc_collection_for_processing(5.,n_control_workers=1)#2)
        sample_len_2 = 5000
        t2 = np.linspace(0,1000,sample_len_2)
        self.omega1 = 1.
        phi1 = 0.
        self.omega2 = .782927
        self.omega3 = 1.2023432
        phi2 = .3422982
        self.col_b.add_object(t2,3. + np.sin(self.omega1*t2+phi1) +
                                  .25*np.sin(self.omega2*t2+phi2),
                                  [.05]*sample_len_2,
                                  0.,0.,'o1')
        self.col_b.add_object(t2,3. + .12*np.sin(self.omega3*t2+phi1) +
                                  .5*np.sin(self.omega2*t2+phi2),
                                  [.03]*sample_len_2,
                                  0.,0.,'o2')
        self.col_b.add_object(t2,3.+.94*np.sin(self.omega2*t2+phi2),
                                  [.05]*sample_len_2,3.,3.99,'o3')
        self.col_b.add_object(t2,[3.5]*sample_len_2,
                                  [.05]*sample_len_2,2.,1.5,'o4')
        self.col_b.add_object(t2,2. + 3.*np.sin(self.omega2*t2+phi2) +
                                  .5*np.sin(self.omega1*t2+phi1),
                                  [.07]*sample_len_2,10000,10000,'o5')
        

    def test_lc_collection_setup(self):
        # Test initialization of the lc_collection_for_processing class
        self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                                dproc.lc_collection_for_processing)

    def test_periodsearch_results_setup(self):
        #Test initialization of the periodsearch_results class
        self.assertIsInstance(dproc.periodsearch_results('1'),
                                  dproc.periodsearch_results)

    def _test_basic_ls_run(self):
        # Test a basic run of the iterative deblending
        self.col_a.run_ls(startp=6.,endp=7.,stepsize=0.0000001,autofreq=False)

        with self.assertRaises(KeyError):
            self.col_a.results['object1']['BLS']
        with self.assertRaises(KeyError):
            self.col_a.results['object2']['LS']
        self.assertEqual(len(self.col_a.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a.results['object1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi,places=6)
        self.assertEqual(len(self.col_a.results['object1']),1)
        self.assertEqual(len(self.col_a.results['object1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_a.results['object2']),0)

        
    def test_blended_ls_run(self):
        self.col_b.run_ls(startp=2.,endp=11.,stepsize=1e-5,autofreq=False)

        # Check o1
        self.assertEqual(len(self.col_b.results[self.col_b.index_dict['o1']]['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_b.results[self.col_b.index_dict['o1']]['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi*self.omega1,places=3)
        self.assertEqual(len(self.col_b.results[self.col_b.index_dict['o1']]['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_b.results[self.col_b.index_dict['o1']]['LS'].blends_info[0]['lsp_dict']['bestperiod'],2*np.pi*self.omega2,places=3)
        self.assertEqual(self.col_b.results[self.col_b.index_dict['o1']]['LS'].blends_info[0]['ID_of_blend'],'o3')

        # Check o2
        self.assertEqual(len(self.col_b.results[self.col_b.index_dict['o2']]['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_b.results[self.col_b.index_dict['o2']]['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi*self.omega3,places=3)
        self.assertEqual(len(self.col_b.results[self.col_b.index_dict['o2']]['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_b.results[self.col_b.index_dict['o2']]['LS'].blends_info[0]['lsp_dict']['bestperiod'],2*np.pi*self.omega2,places=3)
        self.assertEqual(self.col_b.results[self.col_b.index_dict['o2']]['LS'].blends_info[0]['ID_of_blend'],'o3')

        # Check o3
        self.assertEqual(len(self.col_b.results[self.col_b.index_dict['o3']]['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_b.results[self.col_b.index_dict['o3']]['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_b.results[self.col_b.index_dict['o3']]['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2*np.pi.self.omega2,places=3)
        
        # Check o4
        with self.assertRaises(KeyError):
            self.col_b.results['o4']['LS']
        self.assertEqual(len(self.col_b.results['o4'].keys()),0)




if __name__ == '__main__':
    unittest.main()
