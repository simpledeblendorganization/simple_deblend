'''test_data_processing.py - Joshua Wallace - Feb 2019

This tests classes and methods in the data_processing.py file.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
print(os.path.abspath('../src'))
import data_processing as dproc
#import lightcurve_class.single_lc_object


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
        col_a = lc_collection_for_processing(1.,nworkers=1)
        col_a.add_object(lcclass

    def test_lc_collection_setup(self):
        # Test initialization of the lc_collection_for_processing class
        self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                                dproc.lc_collection_for_processing)

    def test_periodsearch_results_setup(self):
        #Test initialization of the periodsearch_results class
        self.assertIsInstance(dproc.periodsearch_results('1'),
                                  dproc.periodsearch_results)
        




if __name__ == '__main__':
    unittest.main()
