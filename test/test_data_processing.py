'''test_data_processing.py - Joshua Wallace - Feb 2019

This tests classes and methods in the data_processing.py file.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
print(os.path.abspath('../src'))
import data_processing as dproc


class test_data_processing(unittest.TestCase):

    # Test initialization of the lc_collection_for_processing class:
    self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                              dproc.lc_collection_for_processing)

