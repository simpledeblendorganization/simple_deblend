'''test_data_processing_ls.py - Joshua Wallace - Feb 2019

This tests classes and methods in the data_processing.py file,
using the Lomb-Scargle method.
'''

import unittest
import sys, os
sys.path.insert(1,os.path.abspath('../src'))
import data_processing as dproc
import numpy as np
import copy


class test_data_processing_initialization(unittest.TestCase):

    def test_lc_collection_setup(self):
        # Test initialization of the lc_collection_for_processing class
        self.assertIsInstance(dproc.lc_collection_for_processing(1.),
                                dproc.lc_collection_for_processing)

    def test_periodsearch_results_setup(self):
        #Test initialization of the periodsearch_results class
        self.assertIsInstance(dproc.periodsearch_results('1'),
                                  dproc.periodsearch_results)


class test_data_processing_error_filterwindows(unittest.TestCase):

    def setUp(self):
        self.col1 = dproc.lc_collection_for_processing(1.,n_control_workers=1)
        length = 1500
        t1 = np.linspace(0,100,length)
        self.col1.add_object(t1,[10.]*length+.3*np.sin(t1),[.5]*length,0.,0.,'1')
        self.col2 = copy.deepcopy(self.col1)
        self.col3 = copy.deepcopy(self.col1)


    def test_window_toolarge(self):
        # Test that snr freq window index size is too large raises error
        with self.assertRaises(ValueError):
            self.col1.run_ls(startp=20.,endp=25.,stepsize=5e-5,autofreq=False,
                             medianfilter=False,
                             freq_window_epsilon_snr=5.,
                             snr_filter_size=1500,snr_threshold=1.)

    def test_window_90percent(self):
        # Test that snr freq window index size is large raises error
        with self.assertRaises(ValueError):
            self.col2.run_ls(startp=20.,endp=25.,stepsize=5e-5,autofreq=False,
                             medianfilter=False,
                             freq_window_epsilon_snr=0.95,
                             snr_filter_size=1500,snr_threshold=1.)


class test_data_processing_ls_sinusoidal_single_signal(unittest.TestCase):

    def setUp(self):
        
        # random number
        rand1 = np.random.RandomState(seed=1844)
        
        ### col_a, very basic, two-object test
        self.col_a = dproc.lc_collection_for_processing(1.,n_control_workers=2)
        sample_len_1 = 3000
        sigma1 = .08
        t1 = np.linspace(0,1200,sample_len_1)
        self.mag_a = 10.
        self.sinamp_a = 1.
        self.fluxamp_a = 10**(-0.4*(self.mag_a-self.sinamp_a)) -\
                       10**(-0.4*(self.mag_a+self.sinamp_a))
        self.col_a.add_object(t1,self.mag_a+self.sinamp_a*np.sin(t1)+\
                              sigma1*rand1.randn(sample_len_1),
                              [sigma1]*sample_len_1,0.,0.,'object1')
        # Modify one value 
        #self.col_a.objects[0].mags[3] =\
        #                self.col_a.objects[0].mags[3] + .001
        self.col_a.add_object(t1,np.array([self.mag_a]*sample_len_1)+sigma1*rand1.randn(sample_len_1),
                                  [sigma1]*sample_len_1,0.5,0,'object2')

        self.col_a2 = copy.deepcopy(self.col_a)

        
        ### col_c, test with some blending
        rand3 = np.random.RandomState(seed=1847)
        self.col_c = dproc.lc_collection_for_processing(5,n_control_workers=2)

        self.mag_c1 = 8.
        self.mag_c2 = 10.
        self.mag_c3 = 4.
        self.amp_c1 = 1.
        self.amp_c2 = 0.5
        self.amp_c3 = 0.1
        self.fluxamp_c1 = 10**(-0.4*(self.mag_c1-self.amp_c1)) -\
                       10**(-0.4*(self.mag_c1+self.amp_c1))
        self.fluxamp_c2 = 10**(-0.4*(self.mag_c2-self.amp_c2)) -\
                       10**(-0.4*(self.mag_c2+self.amp_c2))
        self.fluxamp_c3 = 10**(-0.4*(self.mag_c3-self.amp_c3)) -\
                       10**(-0.4*(self.mag_c3+self.amp_c3))

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
        self.col_c.add_object(t3,self.mag_c1 +\
                              self.amp_c1*np.sin(self.omegac*t3 + phi_c) +\
                              sigmac1*temp1,
                              [sigmac1]*sample_len_3,0.,1.,'c1')
        self.col_c.add_object(t3, self.mag_c2 +
                              self.amp_c2*np.sin(self.omegac*t3 + phi_c) +
                              sigmac2*temp2,
                              [sigmac2]*sample_len_3,1.,2.5,'c2')
        self.col_c.add_object(t3,self.mag_c3 +
                              self.amp_c3*np.sin(self.omegac*t3 + phi_c) +
                                  sigmac3*temp3,
                                  [sigmac3]*sample_len_3,10234.,-2327912.7,'c3')

        
        ### col_d, long-period test 
        rand4 = np.random.RandomState(seed=1853)
        self.col_d = dproc.lc_collection_for_processing(10,n_control_workers=1)
        
        sample_len_4 = 5002
        t4 = np.linspace(0,121.2,sample_len_4)
        self.omegad = 2.*np.pi/30.1107
        sigma4 = .2
        self.mag_d = 5.
        self.amp_d = 0.75
        self.fluxamp_d = 10**(-0.4*(self.mag_d-self.amp_d)) -\
                          10**(-0.4*(self.mag_d+self.amp_d))
        self.col_d.add_object(t4, self.mag_d+
                              self.amp_d*np.sin(self.omegad*t4)+
                              sigma4*rand4.randn(sample_len_4),
                              [sigma4]*sample_len_4,0.,1.0,'d1')
        
        ### col_e, multiple blended objects
        rand5 = np.random.RandomState(seed=1893)
        self.col_e = dproc.lc_collection_for_processing(10.,n_control_workers=2)

        sample_len_5 = 2855
        t5 = np.linspace(0,100.5,sample_len_5)
        self.omegae = 2.*np.pi/3.48208#3.47464
        phi_e = .3
        sigma5 = .1
        self.mag_e = 4.
        self.amp_e1 = 1.
        self.amp_e2 = .4
        self.amp_e3 = .3
        self.amp_e4 = .2
        self.fluxamp_e1 = 10**(-0.4*(self.mag_e-self.amp_e1)) -\
                          10**(-0.4*(self.mag_e+self.amp_e1))
        self.fluxamp_e2 = 10**(-0.4*(self.mag_e-self.amp_e2)) -\
                          10**(-0.4*(self.mag_e+self.amp_e2))
        self.fluxamp_e3 = 10**(-0.4*(self.mag_e-self.amp_e3)) -\
                          10**(-0.4*(self.mag_e+self.amp_e3))
        self.fluxamp_e4 = 10**(-0.4*(self.mag_e-self.amp_e4)) -\
                          10**(-0.4*(self.mag_e+self.amp_e4))
        self.col_e.add_object(t5, self.mag_e+
                              self.amp_e1*np.sin(self.omegae*t5+phi_e)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,0.,0.5,'e1')
        self.col_e.add_object(t5, self.mag_e+
                              self.amp_e2*np.sin(self.omegae*t5+phi_e)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,0.5,0.,'e2')
        self.col_e.add_object(t5, self.mag_e+
                              self.amp_e3*np.sin(self.omegae*t5+phi_e)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,0.7,0.1,'e3')
        self.col_e.add_object(t5, self.mag_e+
                              self.amp_e4*np.sin(self.omegae*t5+phi_e)+
                              sigma5*rand5.randn(sample_len_5),
                              [sigma5]*sample_len_5,3.0,0.9,'e4')
        
                             
    
    def test_basic_ls_run(self):
        # Test a basic run of the iterative deblending
        self.col_a.run_ls(startp=6.,endp=7.,stepsize=0.0000001,autofreq=False,
                          medianfilter=False,
                          freq_window_epsilon_snr=10.,
                          snr_filter_size=50000,snr_threshold=[12.,8.])

        with self.assertRaises(KeyError):
            self.col_a.results['object1']['BLS']
        self.assertEqual(len(self.col_a.results['object1'].keys()),1)
        self.assertEqual(len(self.col_a.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a.results['object1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi,places=6)
        self.assertEqual(len(self.col_a.results['object1']),1)
        self.assertEqual(len(self.col_a.results['object1']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_a.results['object1']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_a,places=5)
        self.assertEqual(len(self.col_a.results['object1']['LS'].blends_info),0)
        


    def test_basic_ls_run_medianfilter(self):
        # Test a basic run of the iterative deblending
        self.col_a2.run_ls(startp=6.,endp=7.,stepsize=0.00001,autofreq=False,
                          medianfilter=True,
                          freq_window_epsilon_mf=10.,
                          median_filter_size=500,
                          freq_window_epsilon_snr=10.,
                          snr_filter_size=500,snr_threshold=[12.,11.])

        with self.assertRaises(KeyError):
            self.col_a2.results['object1']['BLS']
        self.assertEqual(len(self.col_a2.results['object1'].keys()),1)
        self.assertEqual(len(self.col_a2.results['object2'].keys()),0)

        self.assertAlmostEqual(self.col_a2.results['object1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi,places=3)
        self.assertEqual(len(self.col_a2.results['object1']),1)
        self.assertEqual(len(self.col_a2.results['object1']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_a2.results['object1']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_a,places=5)
        self.assertEqual(len(self.col_a2.results['object1']['LS'].blends_info),0)

       
    def test_simple_blended_ls_run(self):
        self.col_c.run_ls(startp=0.5,endp=2.,stepsize=5e-5,autofreq=False,
                          medianfilter=False,
                          freq_window_epsilon_snr=5.,
                          snr_filter_size=1500,snr_threshold=12.)

        # Check c1
        self.assertEqual(len(self.col_c.results['c1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c1']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c1']['LS'].good_periods_info[0]['num_previous_blends'],0)
        self.assertAlmostEqual(self.col_c.results['c1']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_c1,places=4)

        # Check c2
        self.assertEqual(len(self.col_c.results['c2']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_c.results['c2']['LS'].blends_info),1)
        self.assertEqual(self.col_c.results['c2']['LS'].blends_info[0]['ID_of_blend'],'c1')
        self.assertAlmostEqual(self.col_c.results['c2']['LS'].blends_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertAlmostEqual(self.col_c.results['c2']['LS'].blends_info[0]['flux_amplitude'],self.fluxamp_c2,places=2)

        #Check c3
        self.assertEqual(len(self.col_c.results['c3']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_c.results['c3']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_c.results['c3']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegac,places=4)
        self.assertEqual(self.col_c.results['c3']['LS'].good_periods_info[0]['num_previous_blends'],0)
        self.assertAlmostEqual(self.col_c.results['c3']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_c3,places=3)

    
    def test_longperiod(self):
        # Test a long period object
        self.col_d.run_ls(startp=20.,endp=50.,autofreq=False,
                          stepsize=1e-5,medianfilter=False,
                          freq_window_epsilon_snr=2.0,snr_filter_size=200,snr_threshold=100.)

        self.assertEqual(len(self.col_d.results['d1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_d.results['d1']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_d.results['d1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegad,places=1)
        self.assertAlmostEqual(self.col_d.results['d1']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_d,places=2)


    
    def test_multipleblend(self):
        # Test a long period object and also objects with multiple blends
        self.col_e.run_ls(startp=2.5,endp=4.7,autofreq=False,
                          stepsize=1e-5,
                          medianfilter=False,freq_window_epsilon_snr=3.5,
                          snr_filter_size=1000,snr_threshold=12.)

        self.assertEqual(len(self.col_e.results['e1']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_e.results['e1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omegae,places=2)
        self.assertAlmostEqual(self.col_e.results['e1']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp_e1,places=2)
        self.assertEqual(self.col_e.results['e1']['LS'].good_periods_info[0]['significant_blends'],['e2','e3'])
        self.assertEqual(self.col_e.results['e1']['LS'].good_periods_info[0]['num_previous_blends'],0)
        self.assertEqual(len(self.col_e.results['e2']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e3']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e4']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col_e.results['e2']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_e.results['e2']['LS'].blends_info[0]['flux_amplitude'],self.fluxamp_e2,places=3)
        self.assertEqual(len(self.col_e.results['e3']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_e.results['e3']['LS'].blends_info[0]['flux_amplitude'],self.fluxamp_e3,places=3)
        self.assertEqual(len(self.col_e.results['e4']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_e.results['e4']['LS'].blends_info[0]['flux_amplitude'],self.fluxamp_e4,places=3)
        self.assertEqual(self.col_e.results['e2']['LS'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e3']['LS'].blends_info[0]['ID_of_blend'],'e1')
        self.assertEqual(self.col_e.results['e4']['LS'].blends_info[0]['ID_of_blend'],'e1')




    
class test_data_processing_run_sinusoidal_multiple_signals(unittest.TestCase):

    def setUp(self):
        # another random number
        rand2 = np.random.RandomState(seed=1847)
        
        ### col_b, more full test with some blending
        self.col_b = dproc.lc_collection_for_processing(5.,n_control_workers=2)
        sample_len_2 = 5000
        t2 = np.linspace(0,1000,sample_len_2)
        self.omega1 = 1.
        phi1 = 0.
        self.omega2 = 0.582885389#.582927
        self.omega3 = 1.2023432
        phi2 = .3422982
        sigma1 = .01
        self.mag1 = 3.
        self.mag2 = 3.
        self.mag3 = 3.
        self.mag4 = 3.5
        self.mag5 = 2.
        self.amp11 = 1.
        self.amp12 = .1
        self.amp23 = .12
        self.amp22 = .6
        self.amp32 = .94
        self.amp52 = 3.
        self.amp51 = .5

        self.fluxamp11 = 10**(-0.4*(self.mag1-self.amp11)) -\
                          10**(-0.4*(self.mag1+self.amp11))
        self.fluxamp12 = 10**(-0.4*(self.mag1-self.amp12)) -\
                          10**(-0.4*(self.mag1+self.amp12))
        self.fluxamp23 = 10**(-0.4*(self.mag2-self.amp23)) -\
                          10**(-0.4*(self.mag2+self.amp23))
        self.fluxamp22 = 10**(-0.4*(self.mag2-self.amp22)) -\
                          10**(-0.4*(self.mag2+self.amp22))
        self.fluxamp32 = 10**(-0.4*(self.mag3-self.amp32)) -\
                          10**(-0.4*(self.mag3+self.amp32))
        self.fluxamp52 = 10**(-0.4*(self.mag5-self.amp52)) -\
                          10**(-0.4*(self.mag5+self.amp52))
        self.fluxamp51 = 10**(-0.4*(self.mag5-self.amp51)) -\
                          10**(-0.4*(self.mag5+self.amp51))

        self.col_b.add_object(t2,self.mag1 +
                              self.amp11*np.sin(self.omega1*t2+phi1) +
                              self.amp12*np.sin(self.omega2*t2+phi2) +
                              sigma1*rand2.randn(sample_len_2),
                              [sigma1]*sample_len_2,
                              0.,0.,'o1')
        sigma2 = .03
        self.col_b.add_object(t2,self.mag2 +
                              self.amp23*np.sin(self.omega3*t2+phi1) +
                              self.amp22*np.sin(self.omega2*t2+phi2) +
                              sigma2*rand2.randn(sample_len_2),
                              [sigma2]*sample_len_2,
                              0.,0.0001,'o2')
        sigma3 = .05
        self.col_b.add_object(t2,self.mag3+
                              self.amp32*np.sin(self.omega2*t2+phi2) +
                              sigma3*rand2.randn(sample_len_2),
                              [sigma3]*sample_len_2,3.,3.99,'o3')
        sigma4 = .01
        self.col_b.add_object(t2,[self.mag4]*sample_len_2,
                              [sigma4]*sample_len_2,2.,1.5,'o4')
        sigma5 = .07
        self.col_b.add_object(t2,self.mag5 +
                              self.amp52*np.sin(self.omega2*t2+phi2) +
                              self.amp51*np.sin(self.omega1*t2+phi1) +
                              sigma5*rand2.randn(sample_len_2),
                              [sigma5]*sample_len_2,10000,10000,'o5')


        
    def test_blended_ls_run(self):
        self.col_b.run_ls(startp=2.,endp=11.,stepsize=1e-5,autofreq=False,
                          medianfilter=False,freq_window_epsilon_snr=4.,
                          snr_filter_size=1000,snr_threshold=300.)

        # Check o1
        self.assertEqual(len(self.col_b.results['o1']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_b.results['o1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omega1,places=3)
        self.assertAlmostEqual(self.col_b.results['o1']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp11,places=4)
        self.assertEqual(len(self.col_b.results['o1']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_b.results['o1']['LS'].blends_info[0]['lsp_dict']['bestperiod'],2*np.pi/self.omega2,places=5)
        self.assertEqual(self.col_b.results['o1']['LS'].blends_info[0]['ID_of_blend'],'o3')
        self.assertAlmostEqual(self.col_b.results['o1']['LS'].blends_info[0]['flux_amplitude'],self.fluxamp12,places=3)
 

        # Check o2
        self.assertEqual(len(self.col_b.results['o2']['LS'].good_periods_info),1)
        self.assertAlmostEqual(self.col_b.results['o2']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2.*np.pi/self.omega3,places=3)
        self.assertAlmostEqual(self.col_b.results['o2']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp23,places=3)
        self.assertEqual(len(self.col_b.results['o2']['LS'].blends_info),1)
        self.assertAlmostEqual(self.col_b.results['o2']['LS'].blends_info[0]['lsp_dict']['bestperiod'],2*np.pi/self.omega2,places=5)
        self.assertAlmostEqual(self.col_b.results['o2']['LS'].blends_info[0]['flux_amplitude'],self.fluxamp22,places=4)
        self.assertEqual(self.col_b.results['o2']['LS'].blends_info[0]['ID_of_blend'],'o3')

        # Check o3
        self.assertEqual(len(self.col_b.results['o3']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col_b.results['o3']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_b.results['o3']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2*np.pi/self.omega2,places=5)
        self.assertAlmostEqual(self.col_b.results['o3']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp32,places=2)

        
        # Check o4
        with self.assertRaises(KeyError):
            self.col_b.results['o4']['LS']
        self.assertEqual(len(self.col_b.results['o4'].keys()),0)

        # Check o5
        self.assertEqual(len(self.col_b.results['o5'].keys()),1)
        self.assertEqual(len(self.col_b.results['o5']['LS'].good_periods_info),2)
        self.assertEqual(len(self.col_b.results['o5']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col_b.results['o5']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],2*np.pi/self.omega2,places=5)
        self.assertAlmostEqual(self.col_b.results['o5']['LS'].good_periods_info[0]['flux_amplitude'],self.fluxamp52,places=2)
        self.assertAlmostEqual(self.col_b.results['o5']['LS'].good_periods_info[1]['lsp_dict']['bestperiod'],2*np.pi/self.omega1,places=3)
        self.assertAlmostEqual(self.col_b.results['o5']['LS'].good_periods_info[1]['flux_amplitude'],self.fluxamp51,places=2)



class test_data_processing_ls_rrlyrae_signal(unittest.TestCase):

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


    def test_rrblend(self):
        # Test an RRab and a blend
        self.col.run_ls(startp=.2,endp=1.,stepsize=1e-4,autofreq=False,
                        medianfilter=False,freq_window_epsilon_snr=4.,
                        snr_filter_size=400,snr_threshold=15.)

        print(self.col.results['rr1']['LS'].good_periods_info[0]['flux_amplitude'])
        print('slssl')
        print(self.col.results['rr2']['LS'].blends_info[0]['flux_amplitude'])
        self.assertEqual(len(self.col.results['rr1']['LS'].good_periods_info),1)
        self.assertEqual(len(self.col.results['rr1']['LS'].blends_info),0)
        self.assertAlmostEqual(self.col.results['rr1']['LS'].good_periods_info[0]['lsp_dict']['bestperiod'],self.rr_period,places=2)
        self.assertAlmostEqual(self.col.results['rr1']['LS'].good_periods_info[0]['flux_amplitude'],4.1131184e-6,places=4)


        self.assertEqual(len(self.col.results['rr2']['LS'].good_periods_info),0)
        self.assertEqual(len(self.col.results['rr2']['LS'].blends_info),1)
        self.assertEqual(self.col.results['rr2']['LS'].blends_info[0]['ID_of_blend'],'rr1')
        self.assertAlmostEqual(self.col.results['rr2']['LS'].blends_info[0]['flux_amplitude'],3.3142017e-6,places=4)
  

if __name__ == '__main__':
    unittest.main()
