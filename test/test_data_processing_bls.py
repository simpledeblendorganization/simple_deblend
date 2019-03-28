'''test_data_processing_ls.py - Joshua Wallace - Mar 2019

This tests classes and methods in the data_processing.py file,
using the Box-fitting Least Squares method.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc
import numpy as np
import copy



def transit_insert(lc,t,depth,epoch,q,period):
    lc_return = []
    for point, tp in zip(lc,t):
        t_mod = ((tp - epoch) % period)/period
        if t_mod < q/2. or t_mod > (1.-q/2.):
            lc_return.append(point + depth)
        else:
            lc_return.append(point)

    return np.array(lc_return)



"""
class test_data_processing_basic_bls_run(unittest.TestCase):

    def setUp(self):
        # random number
        rand1 = np.random.RandomState(seed=1844)
        
        ### col_a, very basic, two-object test
        self.col_a = dproc.lc_collection_for_processing(1.,n_control_workers=2)
        sample_len_1 = 3000
        sigma1 = .08
        t1 = np.linspace(0,1200,sample_len_1)
        self.period = 2.*np.pi
        self.mag = 10.
        self.depth = .6
        self.fluxamp = 10**(-0.4*self.mag) - 10**(-0.4*(self.mag+self.depth))
        lc_start = [self.mag]*sample_len_1
        import matplotlib.pyplot as plt
        plt.scatter(t1%self.period,transit_insert(lc_start,t1,self.depth,1.5,.05,self.period)+sigma1*rand1.randn(sample_len_1),s=2)
        plt.savefig("temp.pdf")
        self.col_a.add_object(t1,
                              transit_insert(lc_start,t1,self.depth,1.5,.05,self.period)+\
                                sigma1*rand1.randn(sample_len_1),
                              [sigma1]*sample_len_1,0.,0.,'object1')

        self.col_a.add_object(t1,
                              np.array([self.mag]*sample_len_1)+\
                                sigma1*rand1.randn(sample_len_1),
                              [sigma1]*sample_len_1,0.5,0,'object2')

        self.col_a2 = copy.deepcopy(self.col_a)


    def test_basic_ls_run(self):
        # Test a basic run of the iterative deblending
        self.col_a.run_bls(startp=6.,endp=7.,stepsize=0.0000001,autofreq=False,
                          medianfilter=False,
                          freq_window_epsilon_snr=10.,
                           snr_filter_size=50000,snr_threshold=40.)

        with self.assertRaises(KeyError):
            self.col_a.results['object1']['LS']
        with self.assertRaises(KeyError):
            self.col_a.results['object1']['PDM']
        self.assertEqual(len(self.col_a.results['object1'].keys()),1)
        self.assertEqual(len(self.col_a.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a.results['object1']['BLS'].good_periods_info[0]['lsp_dict']['bestperiod'],self.period,places=3)
        self.assertEqual(len(self.col_a.results['object1']),1)
        self.assertEqual(len(self.col_a.results['object1']['BLS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_a.results['object1']['BLS'].good_periods_info[0]['flux_amplitude'],self.fluxamp,places=5)
        self.assertEqual(len(self.col_a.results['object1']['BLS'].blends_info),0)


    def test_basic_ls_run_medianfilter(self):
        # Test a basic run of the iterative deblending
        self.col_a2.run_bls(startp=6.,endp=7.,stepsize=0.00001,autofreq=False,
                          medianfilter=True,
                          freq_window_epsilon_mf=10.,
                          median_filter_size=500,
                          freq_window_epsilon_snr=10.,
                            snr_filter_size=500,snr_threshold=40.)

        with self.assertRaises(KeyError):
            self.col_a2.results['object1']['LS']
        with self.assertRaises(KeyError):
            self.col_a2.results['object1']['PDM']
        self.assertEqual(len(self.col_a2.results['object1'].keys()),1)
        self.assertEqual(len(self.col_a2.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a2.results['object1']['BLS'].good_periods_info[0]['lsp_dict']['bestperiod'],self.period,places=3)
        self.assertEqual(len(self.col_a2.results['object1']),1)
        self.assertEqual(len(self.col_a2.results['object1']['BLS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_a2.results['object1']['BLS'].good_periods_info[0]['flux_amplitude'],self.fluxamp,places=5)
        self.assertEqual(len(self.col_a2.results['object1']['BLS'].blends_info),0)



class test_data_processing_simple_blended_bls_run(unittest.TestCase):

    def setUp(self):
        ### col_c, test with some blending
        rand3 = np.random.RandomState(seed=1847)
        self.col_c = dproc.lc_collection_for_processing(5,n_control_workers=1)#2)

        self.mag_c1 = 8.
        self.mag_c2 = 10.
        self.mag_c3 = 4.
        self.depth_c1 = 1.
        self.depth_c2 = 0.5
        self.depth_c3 = 0.1
        self.fluxamp_c1 = 10**(-0.4*self.mag_c1) -\
                       10**(-0.4*(self.mag_c1+self.depth_c1))
        self.fluxamp_c2 = 10**(-0.4*self.mag_c2) -\
                       10**(-0.4*(self.mag_c2+self.depth_c2))
        self.fluxamp_c3 = 10**(-0.4*self.mag_c3) -\
                       10**(-0.4*(self.mag_c3+self.depth_c3))

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
        self.col_c.add_object(t3,transit_insert([self.mag_c1]*sample_len_3,
                                                t3,self.depth_c1,0.,0.05,
                                                2*np.pi/self.omegac)+\
                              sigmac1*temp1,
                              [sigmac1]*sample_len_3,0.,1.,'c1')
        self.col_c.add_object(t3,transit_insert([self.mag_c2]*sample_len_3,
                                                t3,self.depth_c2,0.,0.05,
                                                2*np.pi/self.omegac)  +
                              sigmac2*temp2,
                              [sigmac2]*sample_len_3,1.,2.5,'c2')
        self.col_c.add_object(t3,transit_insert([self.mag_c3]*sample_len_3,
                                                t3,self.depth_c3,0.,0.05,
                                                2*np.pi/self.omegac) +
                                  sigmac3*temp3,
                                  [sigmac3]*sample_len_3,10234.,-2327912.7,'c3')




    def test_simple_blended_ls_run(self):
        self.col_c.run_bls(startp=0.5,endp=2.,stepsize=5e-5,autofreq=False,
                          medianfilter=False,
                          freq_window_epsilon_snr=5.,
                           snr_filter_size=1500,snr_threshold=[40.,40.,15.])

        # Check c1
        self.assertEqual(len(self.col_c.results['c1']['BLS'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c1']['BLS'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c1']['BLS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c1']['BLS'].good_periods_info[0]['num_previous_blends'],0)
        self.assertAlmostEqual(self.col_c.results['c1']['BLS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_c1,places=4)

        # Check c2
        self.assertEqual(len(self.col_c.results['c2']['BLS'].good_periods_info),0)
        self.assertEqual(len(self.col_c.results['c2']['BLS'].blends_info),1)
        self.assertEqual(self.col_c.results['c2']['BLS'].blends_info[0]['ID_of_blend'],'c1')
        self.assertAlmostEqual(self.col_c.results['c2']['BLS'].blends_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertAlmostEqual(self.col_c.results['c2']['BLS'].blends_info[0]['flux_amplitude'],self.fluxamp_c2,places=2)

        #Check c3
        self.assertEqual(len(self.col_c.results['c3']['BLS'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c3']['BLS'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c3']['BLS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c3']['BLS'].good_periods_info[0]['num_previous_blends'],0)
        self.assertAlmostEqual(self.col_c.results['c3']['BLS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_c3,places=2)



class test_long_period_bls(unittest.TestCase):

    def setUp(self):

        ### col_d, long-period test 
        rand4 = np.random.RandomState(seed=1853)
        self.col_d = dproc.lc_collection_for_processing(10,n_control_workers=1)
        
        sample_len_4 = 5002
        t4 = np.linspace(0,121.2,sample_len_4)
        self.omegad = 2.*np.pi/30.1107
        sigma4 = .03
        self.mag_d = 5.
        self.depth_d = 0.8
        self.fluxamp_d = 10**(-0.4*self.mag_d) -\
                          10**(-0.4*(self.mag_d+self.depth_d))
        self.col_d.add_object(t4, transit_insert([self.mag_d]*sample_len_4,
                                                 t4,self.depth_d,5.6,.03,
                                                 2*np.pi/self.omegad)+
                              sigma4*rand4.randn(sample_len_4),
                              [sigma4]*sample_len_4,0.,1.0,'d1')


    def test_longperiod(self):
        # Test a long period object
        self.col_d.run_bls(startp=20.,endp=50.,autofreq=False,
                          stepsize=1e-5,medianfilter=False,
                          freq_window_epsilon_snr=2.0,snr_filter_size=200,snr_threshold=60.)

        self.assertEqual(len(self.col_d.results['d1']['BLS'].good_periods_info),1)
        self.assertEqual(len(self.col_d.results['d1']['BLS'].blends_info),0)
        self.assertAlmostEqual(self.col_d.results['d1']['BLS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegad,places=1)
        self.assertAlmostEqual(self.col_d.results['d1']['BLS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_d,places=2)
"""


class test_multiple_blends_bls(unittest.TestCase):

    def setUp(self):

        ### col_e, multiple blended objects
        rand5 = np.random.RandomState(seed=1893)
        self.col_e = dproc.lc_collection_for_processing(10.,n_control_workers=1)#2)

        sample_len_5 = 2855
        t5 = np.linspace(0,100.5,sample_len_5)
        self.omegae = 2.*np.pi/3.48208#3.47464
        phi_e = .3
        sigma5 = .1
        self.mag_e = 4.
        self.depth_e1 = 1.
        self.depth_e2 = .4
        self.depth_e3 = .3
        self.depth_e4 = .15
        self.fluxamp_e1 = 10**(-0.4*self.mag_e) -\
                          10**(-0.4*(self.mag_e+self.depth_e1))
        self.fluxamp_e2 = 10**(-0.4*self.mag_e) -\
                          10**(-0.4*(self.mag_e+self.depth_e2))
        self.fluxamp_e3 = 10**(-0.4*self.mag_e) -\
                          10**(-0.4*(self.mag_e+self.depth_e3))
        self.fluxamp_e4 = 10**(-0.4*self.mag_e) -\
                          10**(-0.4*(self.mag_e+self.depth_e4))
        epoch = 6.57
        q = .06
        self.col_e.add_object(t5,transit_insert([self.mag_e]*sample_len_5,
                                                t5,self.depth_e1,epoch,
                                                q,2.*np.pi/self.omegae)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,0.,0.5,'e1')
        self.col_e.add_object(t5,transit_insert([self.mag_e]*sample_len_5,
                                                t5,self.depth_e2,epoch,
                                                q,2.*np.pi/self.omegae)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,0.5,0.,'e2')
        self.col_e.add_object(t5,transit_insert([self.mag_e]*sample_len_5,
                                                t5,self.depth_e3,epoch,
                                                q,2.*np.pi/self.omegae)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,0.7,0.1,'e3')
        self.col_e.add_object(t5,transit_insert([self.mag_e]*sample_len_5,
                                                t5,self.depth_e4,epoch,
                                                q,2.*np.pi/self.omegae)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,3.0,0.9,'e4')

    
    def test_multipleblend(self):
        # Test a long period object and also objects with multiple blends
        self.col_e.run_bls(startp=2.5,endp=4.7,autofreq=False,
                          stepsize=1e-5,
                          medianfilter=False,freq_window_epsilon_snr=3.5,
                           snr_filter_size=1000,snr_threshold=[42.,30.,30.,30])

        self.assertEqual(len(self.col_e.results['e1']['BLS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_e.results['e1']['BLS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegae,places=2)
        self.assertAlmostEqual(self.col_e.results['e1']['BLS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_e1,places=2)
        self.assertEqual(self.col_e.results['e1']['BLS'].good_periods_info[0]['significant_blends'],['e2','e3'])
        self.assertEqual(self.col_e.results['e1']['BLS'].good_periods_info[0]['num_previous_blends'],0)
        self.assertEqual(len(self.col_e.results['e2']['BLS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e3']['BLS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e4']['BLS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e2']['BLS'].blends_info),1)
        self.assertAlmostEqual(self.col_e.results['e2']['BLS'].blends_info[0]['flux_amplitude'],self.fluxamp_e2,places=3)
        self.assertEqual(len(self.col_e.results['e3']['BLS'].blends_info),1)
        self.assertAlmostEqual(self.col_e.results['e3']['BLS'].blends_info[0]['flux_amplitude'],self.fluxamp_e3,places=3)
        self.assertEqual(len(self.col_e.results['e4']['BLS'].blends_info),1)
        self.assertAlmostEqual(self.col_e.results['e4']['BLS'].blends_info[0]['flux_amplitude'],self.fluxamp_e4,places=3)
        self.assertEqual(self.col_e.results['e2']['BLS'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e3']['BLS'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e4']['BLS'].blends_info[0]['ID_of_blend'],'e1')




if __name__ == '__main__':
    unittest.main()
