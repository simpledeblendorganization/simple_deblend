'''test_data_processing.py - Joshua Wallace - Feb 2019

This tests classes and methods in the data_processing.py file.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
print(os.path.abspath('../src'))
import data_processing as dproc
import scipy


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
        self.col_a = dproc.lc_collection_for_processing(1.,nworkers=1)
        sample_len_1 = 100
        t1 = range(sample_len_1)
        self.col_a.add_object(t1,10.+scipy.sin(t1),[.1]*sample_len_1,0.,0.,'1')
        self.col_a.add_object(t1,[10.]*sample_len_1,[.1]*sample_len_1,0.5,0,'2')

    def test_lc_collection_setup(self):
        # Test initialization of the lc_collection_for_processing class
        self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                                dproc.lc_collection_for_processing)

    def test_periodsearch_results_setup(self):
        #Test initialization of the periodsearch_results class
        self.assertIsInstance(dproc.periodsearch_results('1'),
                                  dproc.periodsearch_results)

    def test_basic_run(self):
        # Test a basic run of the iterative deblending
        self.col_a.run_ls(startp=0.5,endp=4.)

        with self.assertRaises(KeyError):
            self.col_a.results['1']['bls']

        #self.assertEqual(self.col_a.results['1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*scipy.pi)
        self.assertEqual(len(self.col_a.results['1']),1)
        self.assertEqual(len(self.col_a.results['1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_a.results['2']),0)
        




if __name__ == '__main__':
    unittest.main()
