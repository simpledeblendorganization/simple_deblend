'''data_processing.py - Joshua Wallace - Feb 2019

This code is where the rubber meets the road: where the light curve
data actually gets fed in and results actually get calculated.
'''

from light_curve_class import lc_objects
import simple_deblend as sdb
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from astrobase.periodbase.zgls import pgen_lsp as ls_p
from astrobase.periodbase.spdm import stellingwerf_pdm as pdm_p
#from astrobase.periodbase.kbls import bls_parallel_pfind as bls_p
import sys
sys.path.insert(0,"/home/jwallace/phs3-workspace/m4/vetting_signals/periodogram_snr_analysis/transit_inject/")
from astrobase_aaa.periodbase.kbls import bls_parallel_pfind as bls_p
import copy
import pickle
import warnings

from os.path import isfile



class lc_collection_for_processing(lc_objects):
    '''This is the "master class", as it were, of this package.  The methods
    of this class are what actually lead to periods being found, checked,
    calculated, removed, etc. and it is the main way to access the
    abilities of the code

    Subclasses light_curve_class.lc_objects

    The initialization takes one argument and one optional argument:
    radius   - the circular radius, in pixels, for objects to be in 
               sufficient proximity to be regarded as neighbors
    n_control_workers - (optional; default None) the number of workers to use in the
               parallel calculation---value of None defaults to 
               multiprocessing.cpu_count
    '''

    def __init__(self,radius_,n_control_workers=None):

        # Initialize the collection of light curves
        lc_objects.__init__(self,radius_)

        # Figure out how many n_control_workers to use
        if not n_control_workers:
            self.n_control_workers = cpu_count()//4 # Default, break down over 4 stars in parallel
        elif n_control_workers > cpu_count():
            print("n_control_workers was greater than number of CPUs, setting instead to " + str(cpu_count))
            self.n_control_workers = cpu_count()
        else:
            self.n_control_workers = n_control_workers

        print("n_control_workers is: " + str(self.n_control_workers))




    def run_ls(self,num_periods=3,
               startp=None,endp=None,autofreq=True,
               nbestpeaks=1,periodepsilon=0.1,stepsize=None,
               sigclip=float('inf'),nworkers=None,
               verbose=False,medianfilter=False,
               freq_window_epsilon_mf=None,
               freq_window_epsilon_snr=None,
               median_filter_size=None,
               snr_filter_size=None,
               snr_threshold=0.,
               fap_baluev_threshold=0.,
               max_blend_recursion=4,
               outputdir="."):
        '''Run a Lomb-Scargle period search

        This takes a number of optional arguments:
        num_periods           - maximum number of periods to search for

        startp                - minimum period of the search

        endp                  - maximum period of the search

        autofreq              - astrobase autofreq parameter, whether to
               automatically determine the frequency grid

        nbestpeaks            - astrobase nbestpeaks parameter, right now
               adjusting this shouldn't change the code at all

        periodepsilon         - astrobase periodepsilon parameter

        stepsize              - astrobase stepsize parameter, if setting
               manual frequency grid

        sigclip               - astrobase sigclip parameter, sigma
               clipping light curve

        nworkers              - astrobase nworkers parameter, None
               value leads to automatic determination

        verbose               - astrobase verbose parameter

        medianfilter          - whether to median filter the periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        median_filter_size    - number of points to include in 
               calculating the median value for median filter

        snr_filter_size       - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust from periodogram SNR, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        fap_baluev_threshold  - threshold value or function for
               counting a signal as robust from sdb.fap_baluev(), can be:
                    single value -- applies to all objects and periods
                    callable -- function of period

        max_blend_recursion   - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''

        # Value checking
        if autofreq and stepsize:
            raise ValueError("autofreq was set to True, but stepsize was given")

        # Set up params dict for the astrobase search
        if autofreq:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'sigclip':sigclip,'verbose':verbose}
        else:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,'verbose':verbose}
        
        # The period search method
        method = 'LS'

        # Call run 
        self.run(method,ls_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 fap_baluev_threshold=fap_baluev_threshold,
                 max_blend_recursion=max_blend_recursion,
                 outputdir=outputdir)

        
    def run_pdm(self,num_periods=3,
                startp=None,endp=None,autofreq=True,
                nbestpeaks=1,periodepsilon=0.1,stepsize=None,
                sigclip=float('inf'),nworkers=None,
                verbose=False,phasebinsize=0.05,medianfilter=False,
                freq_window_epsilon_mf=None,
                freq_window_epsilon_snr=None,
                median_filter_size=None,
                snr_filter_size=None,
                snr_threshold=0.,
                max_blend_recursion=4,
                outputdir="."):
        '''Run a Phase Dispersion Minimization period search

        This takes a number of optional arguments:
        num_periods           - maximum number of periods to search for

        startp                - minimum period of the search

        endp                  - maximum period of the search

        autofreq              - astrobase autofreq parameter, whether to
               automatically determine the frequency grid

        nbestpeaks            - astrobase nbestpeaks parameter, right now
               adjusting this shouldn't change the code at all

        periodepsilon         - astrobase periodepsilon parameter

        stepsize              - astrobase stepsize parameter, if setting
               manual frequency grid

        sigclip               - astrobase sigclip parameter, sigma
               clipping light curve

        nworkers              - astrobase nworkers parameter, None
               value leads to automatic determination

        verbose               - astrobase verbose parameter

        phasebinsize          - astrobase phasebinsize parameter

        medianfilter          - whether to median filter the periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        median_filter_size    - number of points to include in 
               calculating the median value for median filter

        snr_filter_size       - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        max_blend_recursion   - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''

        # Value checking
        if autofreq and stepsize:
            raise ValueError("autofreq was set to True, but stepsize was given")

        # Set up params dict for the astrobase search
        if autofreq:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'sigclip':sigclip,
                      'verbose':verbose,'phasebinsize':phasebinsize}
        else:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'stepsize':stepsize,'sigclip':sigclip,
                      'verbose':verbose,'phasebinsize':phasebinsize}

        # The period search method
        method = 'PDM'
        
        # Call run
        self.run(method,pdm_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 max_blend_recursion=max_blend_recursion,
                 outputdir=outputdir)


    def run_bls(self,num_periods=3,
                startp=None,endp=None,autofreq=True,
                nbestpeaks=1,periodepsilon=0.1,
                nphasebins=None,stepsize=None,
                mintransitduration=0.01,maxtransitduration=0.4,
                sigclip=float('inf'),nworkers=None,
                verbose=False,medianfilter=False,
                freq_window_epsilon_mf=None,
                freq_window_epsilon_snr=None,
                median_filter_size=None,
                snr_filter_size=None,
                snr_threshold=0.,
                spn_threshold=0.,
                max_blend_recursion=3,
                outputdir="."):
        '''Run a Box-fitting Least Squares period search

        This takes a number of optional arguments:
        num_periods           - maximum number of periods to search for

        startp                - minimum period of the search

        endp                  - maximum period of the search

        autofreq              - astrobase autofreq parameter, whether to
               automatically determine the frequency grid

        nbestpeaks            - astrobase nbestpeaks parameter, right now
               adjusting this shouldn't change the code at all

        periodepsilon         - astrobase periodepsilon parameter

        nphasebins            - astrobase nphasebins parameter

        stepsize              - astrobase stepsize parameter, if setting
               manual frequency grid

        mintransitduration    - astrobase mintransitduration parameter,
               the minimum transit duration to search

        maxtransitduration    - astrobase maxtransitduration parameter,
               the maximum transit duration to search

        sigclip               - astrobase sigclip parameter, sigma
               clipping light curve

        nworkers              - astrobase nworkers parameter, None
               value leads to automatic determination

        verbose               - astrobase verbose parameter

        phasebinsize          - astrobase phasebinsize parameter

        medianfilter          - whether to median filter the periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        median_filter_size    - number of points to include in 
               calculating the median value for median filter

        snr_filter_size       - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust, in the periodogram, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        spn_threshold         - threshold value or function for
               counting signal-to-pink-noise as robust, can be:
                    single value -- applies to all objects and periods
                    callable -- function of period

        max_blend_recursion   - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''

        # Value checking
        if (autofreq and nphasebins) or (autofreq and stepsize):
            raise ValueError("autofreq was set to true, but stepsize and/or nphasebins was given")

        # Set params dict for astrobase
        if autofreq:
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'mintransitduration':mintransitduration,
                      'maxtransitduration':maxtransitduration,
                      'sigclip':sigclip,'verbose':verbose}
        else:
            if not stepsize:
                raise ValueError("autofreq is false, but no value given for stepsize")
            if not nphasebins:
                raise ValueError("autofreq is false, but no value given for nphasebins")
            params = {'startp':startp,'endp':endp,'autofreq':autofreq,
                      'nbestpeaks':nbestpeaks,'periodepsilon':periodepsilon,
                      'mintransitduration':mintransitduration,
                      'maxtransitduration':maxtransitduration,
                      'stepsize':stepsize,'nphasebins':nphasebins,
                      'sigclip':sigclip,'verbose':verbose}

        # The period search method
        method = 'BLS'

        # Call run
        self.run(method,bls_p,params,num_periods,nworkers,
                 medianfilter=medianfilter,
                 freq_window_epsilon_mf=freq_window_epsilon_mf,
                 freq_window_epsilon_snr=freq_window_epsilon_snr,
                 median_filter_size=median_filter_size,
                 snr_filter_size=snr_filter_size,snr_threshold=snr_threshold,
                 spn_threshold=spn_threshold,
                 max_blend_recursion=max_blend_recursion,
                 outputdir=outputdir)
        

    def run(self,which_method,ps_func,params,num_periods,nworkers,
            medianfilter=False,freq_window_epsilon_mf=None,
            freq_window_epsilon_snr=None,median_filter_size=None,
            snr_filter_size=None,snr_threshold=0.,spn_threshold=None,
            fap_baluev_threshold=None,
            max_blend_recursion=8,outputdir="."):
        '''Run a given period search method

        which_method  - the name of the period search method being used
        ps_func       - the period search function from astrobase
        params        - params dict to be passed to ps_func
        num_periods   - maximum number of periods to search
        nworkers      - number of child workers per control worker,
                        can be automatically determined

        Optional parameters:

        medianfilter   - whether to perform median filtering of periodogram

        freq_window_epsilon_mf - sets the size of the exclusion area
               in the periodogram for the SNR calculation

        freq_window_epsilon_snr - sets the size of the exclusion area
               in the periodogram for the median filter calculation

        median_filter_size - number of points to include in 
               calculating the median value for median filter

        snr_filter_size    - number of points to include in
               calculating the standard deviation for the SNR

        snr_threshold         - threshold value or function for
               counting a signal as robust, can be:
                    single value -- applies to all objects and periods
                    iterable -- length of number of objects, applies
                                each value to each object
                    callable -- function of period

        spn_threshold         - threshold value or function for
               counting signal-to-pink-noise as robust, can be:
                    single value -- applies to all objects and periods
                    callable -- function of period
                    None     -- ignore SPN calculation entirely

        fap_baluev_threshold  - threshold value or function for
               counting Baluev FAP measure as robust, can be:
                    single value -- applies to all objects and periods
                    callable -- function of period
                    None     -- ignore fap_baluev calculation entirely

        max_blend_recursion - maximum number of blends to try and fit
               out before giving up

        outputdir - directory for where to save the output

        '''


        # Set nworkers
        num_proc_per_run = max(1,cpu_count()//self.n_control_workers)
        if nworkers is None:
            print("\n***")
            print("None value given to nworkers, auto-calculating a value")
            print("\n***")
        else:
            print("\n***")
            print("nworkers value given")
            if nworkers > num_proc_per_run:
                print("Its value, " + str(nworkers) + " is too large given")
                print(" the number of CPUs and number of control processes")
                print("It is being changed to " + str(num_proc_per_run))

            else:
                num_proc_per_run = nworkers
            print("***\n")
        print("Number of worker processes per control process: " + str(num_proc_per_run) + "\n")

        params['nworkers'] = num_proc_per_run

        # Check what snr_threshold is and set up the tasks list accordingly
        if hasattr(snr_threshold,'__len__'):
            if len(snr_threshold) != len(self.objects):
                raise ValueError("The length of snr_threshold is not the same as the length of objects")
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_val,spn_threshold,
                              fap_baluev_threshold,
                              max_blend_recursion,
                              num_proc_per_run,outputdir)
                             for o, snr_val in zip(self.objects,snr_threshold)]
        elif callable(snr_threshold): # If a callable thing of some kind
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_threshold,spn_threshold,
                              fap_baluev_threshold,
                              max_blend_recursion,
                              num_proc_per_run,outputdir)
                             for o in self.objects]
        else:
            running_tasks = [(o,which_method,ps_func,params,num_periods,
                              medianfilter,freq_window_epsilon_mf,
                              freq_window_epsilon_snr,median_filter_size,
                              snr_filter_size,snr_threshold,spn_threshold,
                              fap_baluev_threshold,
                              max_blend_recursion,
                              num_proc_per_run,outputdir)
                             for o in self.objects]

        #####Temporary change:
        if True:
            # Remove the task if the output already exists

            running_tasks = [r for r in running_tasks if \
                             (not isfile(outputdir + "/ps_" + str(r[0].ID) +\
                                   "_" + which_method + "_goodperiod.pkl")) and#\
                              (not isfile(outputdir + "/ps_" + str(r[0].ID) +\
                                   "_" + which_method + "_blends.pkl"))]

        #running_tasks = [r for r in running_tasks if r[0].ID in ['6045503331994968320']]#['6045466193420246784']]#['6045466193420210816']]

        #running_tasks = [r for r in running_tasks if r[0].ID in ['6045466189124449792', '6045466189124452096', '6045466189124452736', '6045466326563433472', '6045466326563434240', '6045466330858780800', '6045466330858780928', '6045466330858802176', '6045466330858825728', '6045466330858826624', '6045466330858826752', '6045466330859495168', '6045466360918985472', '6045466360923184896', '6045466360923189760', '6045466360927378688', '6045466365218193408', '6045466365218193536', '6045466365218248704', '6045466365218292352', '6045466395287117184', '6045466399577821440', '6045466429642652928', '6045466468297698944', '6045466502657755904', '6045466502657794944', '6045466502657799808', '6045466502657811968', '6045466532721855488', '6045466532721860096', '6045466537017357952', '6045477940156650880', '6045478696063803648', '6045478386835793280', '6045477905795228288', '6045477905795229184', '6045478318112094976', '6045466537017455104', '6045466537017472000', '6045466567077206656', '6045466567081582080', '6045466605737643776', '6045466640096636160', '6045466640096637568', '6045466670165027328', '6045466708816350976', '6045477424753241088', '6045477665271538432', '6045477905789540992', '6045477905795226368', '6045477940149305344', '6045477940149305472', '6045477940149305728', '6045477940154970240', '6045477940154970368', '6045477944452725760', '6045477944452757760', '6045477944452783616', '6045478043228480768', '6045478043228496640', '6045478047532223360', '6045478215027328128', '6045478219330897920', '6045478219330898048', '6045478249387075200', '6045478318107742848', '6045478318113811072', '6045478352466285568', '6045478352471840128', '6045478352471840640', '6045478356769574784', '6045478356769634944', '6045478391129472768', '6045478421185791872', '6045478524266248192', '6045478524272305024', '6045478524274779392', '6045478558624813440', '6045478558624847488', '6045478558632073472', '6045478562928279168', '6045478562928310272', '6045478562928340480', '6045478597288019584', '6045478661704018560', '6045478661704021888', '6045478661704076672', '6045478666007662336', '6045478696063836416', '6045478696070885248', '6045478700367364096', '6045478730423560832', '6045478730424946048', '6045478730430703104', '6045478730433208832', '6045478734727039104', '6045478764783310464', '6045478764783374208', '6045478764783390464', '6045478764790478592', '6045478799150146688', '6045478936581824896', '6045478936581831168', '6045479005301362176', '6045479039661159424', '6045479043964809728', '6045479074027775232', '6045479074027782656', '6045479108380709760', '6045479112684335104', '6045479181403854848', '6045479417618350848', '6045479451985084160', '6045479589417172992', '6045479658136727168', '6045479726856129920', '6045479829935383680', '6045479864295229056', '6045479864295229312', '6045479864296167040', '6045479933014689664', '6045480693232431360', '6045482200757207808', '6045482853592456704', '6045488969625293312', '6045489003985068288', '6045489003985074176', '6045489008290222080', '6045489038344755968', '6045489038344757376', '6045489038344757504', '6045489072704514944', '6045489072704515072', '6045489077009760896', '6045489107064235008', '6045489107064238720', '6045489111369530880', '6045489175783785600', '6045489244504616704', '6045489278863093632', '6045489278863124608', '6045489313222731520', '6045489347582518656', '6045489381942287744', '6045489794267192704', '6045489862978578304', '6045489897338339968', '6045489897338340864', '6045489897346116736', '6045489936003359232', '6045489936003360128', '6045489936003364480', '6045489936003364736', '6045490103496970368', '6045490103496970496', '6045490107802132096', '6045490206576177792', '6045490275301101440', '6045490618893154944', '6045490687612596864', '6045490687617964416', '6045490790691916928', '6045490790691917312', '6045490825051658624', '6045490893771170560', '6045501957606694016', '6045501957606730880', '6045501957606731392', '6045501957606738048', '6045501957613991808', '6045501991966471680', '6045501991966472064', '6045501996271737856', '6045502026326261120', '6045502026326266112', '6045502026333452800', '6045502060685987456', '6045502064991291648', '6045502095053034752', '6045502163765297664', '6045502198125052544', '6045502232484715008', '6045502232490018304', '6045502232491819520', '6045502232491836160', '6045502232494258176', '6045502236790001536', '6045502301204217216', '6045502301204248960', '6045502301213734144', '6045502301213737728', '6045502335563963904', '6045502335564002944', '6045502473003049216', '6045502473003060352', '6045502541722480768', '6045502541722509952', '6045502576082225280', '6045502576082238464', '6045502614747130112', '6045502679159938560', '6045502782240663552', '6045503057118542208', '6045503061423822464', '6045503091478302592', '6045503091485234944', '6045503125838013312', '6045503125847460352', '6045503228917299456', '6045503228917314048', '6045503263277056000', '6045503469435509504','6045465639362931584', '6045465639363028608', '6045465643664656256', '6045466189124456448', '6045466193420167296', '6045466193420186368', '6045466193420196608', '6045466193420224768', '6045466193420273920', '6045466193420582528', '6045466326563431424', '6045466326563432448', '6045466326563438464', '6045466330858789376', '6045466330859491072', '6045466365218195328', '6045466429642654720', '6045466429644559360', '6045466433938150528', '6045466463998200960', '6045466464002400512', '6045466468297774720', '6045466468297825024', '6045466498362112896', '6045466502657803264', '6045466532721857024', '6045466532721858048', '6045466567081583488', '6045466571377348864', '6045466571377393920', '6045466571377396864', '6045466605737162624', '6045466640096752640', '6045466640097159296', '6045466640097175040', '6045466670160826496', '6045466670160827648', '6045466670162729344', '6045466704524764160', '6045466708816308992', '6045466708816661888', '6045466738874738944', '6045466743176092672', '6045476565759710592', '6045476600119453440', '6045476634479183488', '6045476634479188480', '6045476634486343168', '6045476634489098880', '6045476638782998528', '6045477252955831552', '6045477356033681152', '6045477394697352192', '6045477459113058048', '6045477532136361856', '6045477630911716992', '6045477635215605888', '6045477944452703488', '6045477978812648576', '6045478043228497024', '6045478043235649408', '6045478047532096640', '6045478077588248832', '6045478077588256000', '6045478077595420288', '6045478180667514240', '6045478184971102208', '6045478184971231360', '6045478215027308544', '6045478283746872064', '6045478318107741440', '6045478318113833344', '6045478322409875328', '6045478356769586304', '6045478356769607936', '6045478356769617280', '6045478391129551488', '6045478391129564160', '6045478391129564544', '6045478425489317760', '6045478489905360384', '6045478524265081344', '6045478524270536320', '6045478631647842688', '6045478696070882304', '6045478700367408256', '6045478730423545984', '6045479005301398400', '6045479280179418624', '6045479451983450240', '6045479525001272448', '6045482887958839040', '6045489038344767360', '6045489077009761024', '6045489107064264576', '6045489141424041600', '6045489278872986880', '6045489656820248320', '6045490137856618624', '6045490210881403904', '6045490240935944192', '6045490515813875200', '6045490550173615744', '6045490687612631936', '6045501957616360448', '6045502030631545856', '6045502060686019840', '6045502369923755776', '6045502511667851264', '6045503263277059584']]

        # This is LS good periods
        running_tasks = [r for r in running_tasks if r[0].ID in ['6045466193420213760','6045466193420246784','6045466193420272896','6045466326563438464','6045466330858880896','6045466360923184896','6045466330858684416','6045466330858780800','6045466330858825728','6045466395287117184','6045466399577821440','6045466399578084608','6045466399578031616','6045466429642654848','6045466365219206400','6045466395277274240','6045466395282930048','6045466429642651264','6045466429642653696','6045466429642660224','6045466429642664704','6045466429644559360','6045466433938050944','6045466433938695936','6045466433938720384','6045466433938707456','6045466464002403712','6045466433938000512','6045466433938052608','6045466433938719872','6045466464002402048','6045466464002401536','6045466464004363136','6045466468297698944','6045466502657669632','6045466502657717632','6045466502657720576','6045466502657791104','6045466502657794944','6045466468297970176','6045466498362111872','6045466498362115968','6045466498362113280','6045466502657651968','6045489313222731520','6045490687612591872','6045466532721857024','6045489867283826560','6045479761215836928','6045490313960637696','6045503194557549184','6045478219330898048','6045490069137152128','6045477944452703488','6045489175783785600','6045478008876110336','6045502442948517248','6045479726856129920','6045502782240663552','6045489248808504832','6045466537017472000','6045466537017501056','6045466537017528192','6045466537017536256','6045466567077213184','6045466567081582080','6045466537017447808','6045466537017475968','6045466567081579520','6045466537017501184','6045466537018163712','6045466571377396864','6045466193420196608','6045466601443125120','6045466601435691776','6045466571377421952','6045466571377268480','6045466601441330560','6045466571377288064','6045466571377307776','6045466571377348864','6045466601441331200','6045490756332138496','6045502507362712576','6045466605737065856','6045489038352711936','6045503091478302592','6045466189124452096','6045478562928334464','6045478180673128576','6045490142161872384','6045503400722722304','6045491306097565440','6045502369923718016','6045479593721045248','6045502511667851264','6045489244503326592','6045490137856639360','6045489656820248320','6045489381952202880','6045490515813918848','6045466605737645952','6045466605737643776','6045466635801082624','6045466635801084416','6045466635801084032','6045466640096710016','6045466640096636160','6045466640097159296','6045466674456457984','6045466670155206016','6045466704522394240','6045466708816655488','6045466708816661888','6045466670160826496','6045466670165027328','6045466704514927872','6045466704524764160','6045466708816660992','6045466708816662272','6045466738882175744','6045466738882185472','6045476570064046720','6045476600119453440','6045476634479183488','6045476634479188480','6045476634486315264','6045476634486334464','6045476634486343168','6045477252955831552','6045477321674001152','6045477390393498112','6045477424753216000','6045477424753231232','6045477424754534272','6045477356033734656','6045477390394796160','6045477424753234560','6045477527839378176','6045477356033710976','6045477356040755328','6045477360337539072','6045477390393461888','6045477394697338240','6045466193420210688','6045477424753241088','6045477459113005696','6045477459113057792','6045477463416862080','6045477532136361856','6045465639363028608','6045477459113005824','6045466159060528128','6045477527832499840','6045482205060929024','6045490485759243392','6045490691917770368','6045478425489260416','6045490378382387840','6045502232491819520','6045503263277068672','6045490721972380544','6045478700367416832','6045502099351007104','6045479280179372288','6045479417618335360','6045477665271537280','6045477665271552128','6045477733991034240','6045477733991037568','6045477807013764096','6045477630911708288','6045477630911716608','6045477630911716992','6045477630912972160','6045477665271477632','6045477665271530496','6045477871429793536','6045479933014689664','6045478421187031424','6045482716153388800','6045486392644986752','6045502576082238592','6045503233222523392','6045478562928331520','6045478970947104256','6045479074027774720','6045503297636808192','6045503125838040320','6045478597288019584','6045502816600484352','6045502850960169728','6045479108387563648','6045479039661159168','6045502473003049216','6045478524272305024','6045477875733328640','6045477875733369600','6045477905795223552','6045477905795223936','6045477905795225600','6045477905795226368','6045477905795229184','6045477905795228288','6045477910092900096','6045477910092926848','6045477910092936832','6045502301204217216','6045478730423560832','6045478318112094976','6045478661704076672','6045478352466285568','6045478283746872064','6045478386835793280','6045490206576177792','6045478730423545984','6045478558624847488','6045489278863124608','6045502679159938560','6045478283753926656','6045477944452725760','6045478592984465536','6045478391129551488','6045478215027328512','6045501957606730880','6045490893771170560','6045478696063803648','6045479074027782656','6045478661704021888','6045478043228480768','6045479829935383680','6045478700367364096','6045477910093016832','6045477940149305472','6045478455545547264','6045501991966471680','6045477940156650880','6045478455545547008','6045478013172383104','6045489111370900608','6045477940149305728','6045478356769597440','6045478833502877568','6045501996271732096','6045502095053034752','6045478249387075200','6045502232491790464','6045479864295135488','6045490790691916928','6045490069144770688','6045489862988429568','6045489966065554432','6045478764783390464','6045478322409804928','6045501991966472064','6045489862978577280','6045489347582518656','6045478592985658240','6045478219330897920','6045479005301388032','6045489111369530880','6045502576082238464','6045489072704514944','6045478215027329408','6045490039083462656','6045490000417622528','6045478562928253312','6045490760637273984','6045478077595440896','6045478150611534336','6045490073442351488','6045479005301410048','6045478421185791744','6045479589424132992','6045478558632073472','6045501991966469120','6045489794267192704','6045502030631543936','6045489656830112384','6045479520702923008','6045479555064323456','6045478696070885248','6045477944452783616','6045489111369553024','6045490275295701632','6045478455545545984','6045490481454120320','6045479525001272448','6045482819232694272','6045489622460554880','6045479451985084160','6045502335564002944','6045489450661770752','6045490687612629376','6045478524265089664','6045479829935385472','6045490313960633728','6045489794259316736','6045490210881417344','6045478597288038656','6045489038346708608','6045489381942276352','6045478116252277760','6045490348320393472','6045478043228496640','6045478936581825536','6045478833502855424','6045488969625336064','6045489111369540224','6045466193420197376','6045490344015187584','6045490240941355392','6045502232494258176','6045465643664656256','6045479108380709760','6045489003985068288','6045479280179418624','6045490382680146304','6045503331996483328','6045489278863130496','6045477940154969984','6045490721972381952','6045478249387075456','6045502301204249216','6045503160207200768','6045479181403854848','6045489210143561216','6045502163765297792','6045478696070882304','6045502301213737728','6045482166397396608','6045489656820299392','6045490893771191040','6045502404283507072','6045479005306844160','6045478288050504832','6045478666007606784','6045503091478311808','6045502541722509952','6045479761222524800','6045477940149305344','6045479142740293120','6045479692496346624','6045489381942244224','6045477978812645248','6045479658136680576','6045478867862303360','6045479456281704704','6045490794997042688','6045490275303203456','6045489897338339968','6045478077593909120','6045489588100724352','6045489381942257152','6045478249394134272','6045490069137153792','6045503228917313024','6045502369923721728','6045478146314882048','6045479280179405568','6045489077009777408','6045479490641474688','6045478111949301760','6045502095055317888','6045489145729281920','6045478184971122560','6045478494208618752','6045490137856587136','6045502232491836160','6045482166406982016','6045489381942287744','6045503061423822464','6045490825051658624','6045489523686333056','6045503400716012160','6045490107802132096','6045479829935383936','6045503332001700736','6045502473003026304','6045489553741142784','6045489248808499200','6045490790691917312','6045478391129564544','6045478077588256000','6045503370661519872','6045489450661819136','6045478489905371392','6045490618893154944','6045490481459537024','6045478288050537088','6045502477308245248','6045489828619162368','6045478352471842688','6045478077588281216','6045489072710262144','6045479589417177856','6045502095045736576','6045479490641487232','6045477978812617984','6045478906525813248','6045502369923725568','6045479280179364608','6045489936003399168','6045502202430272640','6045479623776884096','6045503469435509504','6045490515813895680','6045490756337439616','6045478837806360064','6045503336301809280','6045503228917327616','6045501991971871744','6045490790691917184','6045482853592463488','6045502404292971904','6045503228917313920','6045478562928279168','6045502202430247168','6045478425489257856','6045490893771135232','6045502850965365760','6045479520697649536','6045489351887565824','6045503160197808000','6045490240943287168','6045479005301338368','6045479009605089920','6045488973930526464','6045490687612631936','6045479417618350848','6045502610441959680','6045490309655390464','6045490034787152640','6045490687612592000','6045478902222057216','6045478936581824896','6045477978812598016','6045478318112098176','6045478322409895168','6045490313960586240','6045478215027328128','6045502129405544704','6045478764790471040','6045478936581835648','6045478936581831168','6045477940149327488','6045488935265572352','6045490004722833664','6045489936003409152','6045478562928334720','6045478661711047168','6045478249387053824','6045503405021289344','6045502060686010880','6045502404283543296','6045490618893152512','6045489622460547840','6045503164503083264','6045482887952209408','6045478524274779392','6045479486337839744','6045489725539795584','6045478180668783616','6045490520119836032','6045489931698086528','6045502919679709184','6045479043964813824','6045477944452778880','6045489416302016768','6045490721972381824','6045478494208746752','6045489695486071936','6045490103496944512','6045479589417173248','6045479142740359680','6045488866546089344','6045490348320383872','6045479043964796288','6045489661125412864','6045478764788702592','6045502850960183552','6045478184971151744','6045490481463833984','6045478494208706176','6045502095053026048','6045478184971231360','6045490687612596864','6045502030631545856','6045479555057384192','6045477944452663424','6045478666007658368','6045478631647785600','6045478489905333120','6045478489912671232','6045477940154970240','6045489210149219328','6045502026333452800','6045466189124451328','6045477978812545536','6045479108386059520','6045479658136658048','6045479456281716480','6045479043964799488','6045479417618360448','6045479898654924800','6045478562928310272','6045490137864139008','6045479417618335488','6045479593721045120','6045479864301939840','6045489244503313408','6045489210143594752','6045479005301398400','6045490550173615744','6045466193420186368','6045489656820311552','6045502266844484864','6045489175783801600','6045478459849650560','6045479417618302336','6045479520704491776','6045502026326261120','6045479074020892800','6045490034777429248','6045502232490018304','6045489588100772736','6045502163765297664','6045503332003272320','6045502301209484416','6045503057118542848','6045479864295229312','6045479868599047424','6045490142161902464','6045478421185902592','6045478425489262080','6045478730430687872','6045503336301801856','6045479834238953216','6045490210881412736','6045477978812538624','6045490000417627904','6045478249387063680','6045478764783374208','6045478872166110848','6045503228917314048','6045489519381327744','6045502541722480768','6045503160197830912','6045503332001701248','6045490313960599808','6045490275295636864','6045479005301338496','6045479108380732800','6045490893776415360','6045479005308347776','6045490240935944192','6045490756332136448','6045489072704539264','6045478249387026176','6045478322409848576','6045479555057403776','6045490275295636992','6045490447094366720','6045478077595417984','6045503336301788928','6045478528568571264','6045503228917299456','6045490073442351744','6045502202430251776','6045479486343185152','6045478867862295808','6045490721972365696','6045502541722473344','6045478047532223360','6045502133710739072','6045478592984464384','6045479177100097920','6045478322409882240','6045490515813926272','6045479039668112768','6045477944452757760','6045479486337887232','6045479726856129792','6045489111370898688','6045478180667545600','6045489210153510784','6045489867283823616','6045502198125038720','6045502614747119104','6045479348898912512','6045478867862292736','6045489416302028544','6045503331994968320']]


        # Start the run
        print("**************************************")
        print("******")
        print("******       Starting " + which_method + " run")
        print("******")
        print("**************************************")
        with ProcessPoolExecutor(max_workers=self.n_control_workers) as executor:
            er = executor.map(self._run_single_object,running_tasks)

        # Collect the results
        pool_results = [x for x in er]

        for result in pool_results:
            if result:
                self.results[result.ID][which_method] = result


    def _run_single_object(self,task):
        ''' Used to run the code for just a single object,
        included in this way to make the code parallelizable.

        task - the task passed from self.run, see that method
               for definition
        '''

        # Extract parameters from task
        (object,which_method,ps_func,params,num_periods,
         medianfilter,freq_window_epsilon_mf,freq_window_epsilon_snr,
         median_filter_size,snr_filter_size,snr_threshold,spn_threshold,
         fap_baluev_threshold,
         max_blend_recursion,nworkers,outputdir) = task

        # Value checking
        if not medianfilter:
            if freq_window_epsilon_mf is not None:
                warnings.warn("medianfilter is False, but freq_window_epsilon_mf is not None, not using medianfilter")
            if median_filter_size is not None:
                warnings.warn("medianfilter is False, but median_filter_size is not None, not using median filter")

        if spn_threshold and which_method.lower() != 'bls':
            raise ValueError("spn_threshold has a value, but the method is not BLS")

        if fap_baluev_threshold and which_method.lower() != 'ls':
            raise ValueError("fap_baluev has a value, but the method is not LS,")


        # Collect the neighbor light curves
        neighbor_lightcurves = {neighbor_ID:(self.objects[self.index_dict[neighbor_ID]].times,
                                     self.objects[self.index_dict[neighbor_ID]].mags,
                                     self.objects[self.index_dict[neighbor_ID]].errs) for neighbor_ID in object.neighbors}

        # Create a place to store the results
        results_storage = periodsearch_results(object.ID)

        # Start iterating
        yprime = object.mags
        while len(results_storage.good_periods_info) < num_periods:
            # Try the iterative deblend
            yprime = sdb.iterative_deblend(object.times,yprime,object.errs,
                                           neighbor_lightcurves,ps_func,
                                           results_storage,
                                           which_method,
                                           function_params=params,
                                           nharmonics_fit=7,
                                           ID=str(object.ID),
                                           medianfilter=medianfilter,
                                           freq_window_epsilon_mf=freq_window_epsilon_mf,
                                           freq_window_epsilon_snr=freq_window_epsilon_snr,
                                           window_size_mf=median_filter_size,
                                           window_size_snr=snr_filter_size,
                                           snr_threshold=snr_threshold,
                                           spn_threshold=spn_threshold,
                                           fap_baluev_threshold=fap_baluev_threshold,
                                           max_blend_recursion=max_blend_recursion,
                                           nworkers=nworkers)
            if yprime is None: # No more results to be had
                break

        # Return based on whether we have info to return
        #if len(results_storage.good_periods_info) > 0 or\
        #        len(results_storage.blends_info) > 0:
        if True:
            # Save as we go
            with open(outputdir + "/ps_" + str(object.ID) + "_" +\
                          which_method + "_goodperiod.pkl","wb") as f:
                pickle.dump(results_storage.good_periods_info,f)
            with open(outputdir + "/ps_" + str(object.ID) + "_" +\
                          which_method + "_blends.pkl","wb") as f:
                pickle.dump(results_storage.blends_info,f)
        if len(results_storage.good_periods_info) > 0 or\
            len(results_storage.blends_info) > 0:         #####templines
            # And return
            return results_storage

        else:
            return None



    def save_periodsearch_results(self,outputdir):
        '''Method used to save the results

        outputdir  - the directory to which to save the results
        '''
        
        print("Saving results...")

        # Loop over the objects
        for k in self.results.keys():
            # Loop over the period search methods
            for k2 in self.results[k].keys():
                r = self.results[k][k2]
                with open(outputdir + "/ps_" + r.ID + "_" + k2 + "_goodperiod.pkl","wb") as f:
                        pickle.dump(r.good_periods_info,f)
                with open(outputdir + "/ps_" + r.ID + "_" + k2 + "_blends.pkl","wb") as f:
                        pickle.dump(r.blends_info,f)


          

            

class periodsearch_results():
    '''A container to store the results of the above
    period search

    The initialization takes one argument and two optional arguments:
    ID                       - ID of object being stored

    count_neighbor_threshold - flux amplitude ratio needs to be at least
                   this for an object to store a blended neighbor's info

    stillcount_blend_factor - flux amplitude ratio needs to be at least this
                   when the object has a lower amplitude for this to 
                   still count as a period for the object
    '''

    def __init__(self,ID,count_neighbor_threshold=0.25,
                 stillcount_blend_factor=0.9):
        self.ID = ID
        self.good_periods_info = []
        self.blends_info = []
        self.count_neighbor_threshold=count_neighbor_threshold
        self.stillcount_blend_factor=stillcount_blend_factor


    def add_good_period(self,lsp_dict,times,mags,errs,period,snr_value,
                        flux_amplitude,significant_blends,
                        ff_params,
                        notmax=False,s_pinknoise=None,
                        fap_baluev=None,
                        ignore_blend=False):
        '''add a good period for the object

        lsp_dict   - the astrobase lsp_dict
        times      - light curve times
        mags       - light curve magnitudes
        errs       - light curve errors
        period     - the associated period
        snr_value  - value of the periodogram SNR
        flux_amplitude - flux amplitude
        significant_blends - neighbors with flux amplitudes above
                             self.count_neighbor_threshold
        ffparams - Fourier fit parameters for the *current* LC
        notmax     - if the object does not have the maximum
                     flux amplitude but is greater than
                     self.stillcount_blend_factor
        s_pinknoise -signal to pink noise value, only for BLS
        ignore_blend - ID of blend being ignore if its being
                       ignored, False otherwise
        '''
        dict_to_add = {'lsp_dict':lsp_dict,
                       'period':period,'times':times,
                       'mags':mags,'errs':errs,
                       'snr_value':snr_value,
                       'flux_amplitude':flux_amplitude,
                       'num_previous_blends':len(self.blends_info),
                       'significant_blends':significant_blends,
                       'not_max':notmax,'ignore_blend':ignore_blend,
                       'ff_params':ff_params}
        if s_pinknoise is not None:
            dict_to_add['s_pinknoise'] = s_pinknoise
        if fap_baluev is not None:
            dict_to_add['fap_baluev'] = fap_baluev
        self.good_periods_info.append(dict_to_add)

    def add_blend(self,lsp_dict,times,mags,errs,neighbor_ID,
                  period,snr_value,
                  flux_amplitude,ff_params,s_pinknoise=None,
                  fap_baluev=None):
        '''add info where the object is blended with another object,
        that object being determined as the variability source

        lsp_dict   - the astrobase lsp_dict
        times      - light curve times
        mags       - light curve magnitudes
        errs       - light curve errors
        neighbor_ID - ID of the variability source
        period     - the associated period
        snr_value  - value of the periodogram SNR
        flux_amplitude - flux amplitude
        ffparams - Fourier fit parameters for the *current* LC
        s_pinknoise -signal to pink noise value, only for BLS
        '''
        dict_to_add = {'lsp_dict':lsp_dict,
                       'ID_of_blend':neighbor_ID,
                       'period':period,
                       'snr_value':snr_value,
                       'flux_amplitude':flux_amplitude,
                       'num_previous_signals':len(self.good_periods_info),
                       'times':times,'mags':mags,'errs':errs,
                       'ff_params':ff_params}
        if s_pinknoise is not None:
            dict_to_add['s_pinknoise'] = s_pinknoise
        if fap_baluev is not None:
            dict_to_add['fap_baluev'] = fap_baluev
        self.blends_info.append(dict_to_add)


