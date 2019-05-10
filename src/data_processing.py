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

        running_tasks = [r for r in running_tasks if r[0].ID in ['6045466189124449792', '6045466189124452096', '6045466189124452736', '6045466326563433472', '6045466326563434240', '6045466330858780800', '6045466330858780928', '6045466330858802176', '6045466330858825728', '6045466330858826624', '6045466330858826752', '6045466330859495168', '6045466360918985472', '6045466360923184896', '6045466360923189760', '6045466360927378688', '6045466365218193408', '6045466365218193536', '6045466365218248704', '6045466365218292352', '6045466395287117184', '6045466399577821440', '6045466429642652928', '6045466468297698944', '6045466502657755904', '6045466502657794944', '6045466502657799808', '6045466502657811968', '6045466532721855488', '6045466532721860096', '6045466537017357952', '6045477940156650880', '6045478696063803648', '6045478386835793280', '6045477905795228288', '6045477905795229184', '6045478318112094976', '6045466537017455104', '6045466537017472000', '6045466567077206656', '6045466567081582080', '6045466605737643776', '6045466640096636160', '6045466640096637568', '6045466670165027328', '6045466708816350976', '6045477424753241088', '6045477665271538432', '6045477905789540992', '6045477905795226368', '6045477940149305344', '6045477940149305472', '6045477940149305728', '6045477940154970240', '6045477940154970368', '6045477944452725760', '6045477944452757760', '6045477944452783616', '6045478043228480768', '6045478043228496640', '6045478047532223360', '6045478215027328128', '6045478219330897920', '6045478219330898048', '6045478249387075200', '6045478318107742848', '6045478318113811072', '6045478352466285568', '6045478352471840128', '6045478352471840640', '6045478356769574784', '6045478356769634944', '6045478391129472768', '6045478421185791872', '6045478524266248192', '6045478524272305024', '6045478524274779392', '6045478558624813440', '6045478558624847488', '6045478558632073472', '6045478562928279168', '6045478562928310272', '6045478562928340480', '6045478597288019584', '6045478661704018560', '6045478661704021888', '6045478661704076672', '6045478666007662336', '6045478696063836416', '6045478696070885248', '6045478700367364096', '6045478730423560832', '6045478730424946048', '6045478730430703104', '6045478730433208832', '6045478734727039104', '6045478764783310464', '6045478764783374208', '6045478764783390464', '6045478764790478592', '6045478799150146688', '6045478936581824896', '6045478936581831168', '6045479005301362176', '6045479039661159424', '6045479043964809728', '6045479074027775232', '6045479074027782656', '6045479108380709760', '6045479112684335104', '6045479181403854848', '6045479417618350848', '6045479451985084160', '6045479589417172992', '6045479658136727168', '6045479726856129920', '6045479829935383680', '6045479864295229056', '6045479864295229312', '6045479864296167040', '6045479933014689664', '6045480693232431360', '6045482200757207808', '6045482853592456704', '6045488969625293312', '6045489003985068288', '6045489003985074176', '6045489008290222080', '6045489038344755968', '6045489038344757376', '6045489038344757504', '6045489072704514944', '6045489072704515072', '6045489077009760896', '6045489107064235008', '6045489107064238720', '6045489111369530880', '6045489175783785600', '6045489244504616704', '6045489278863093632', '6045489278863124608', '6045489313222731520', '6045489347582518656', '6045489381942287744', '6045489794267192704', '6045489862978578304', '6045489897338339968', '6045489897338340864', '6045489897346116736', '6045489936003359232', '6045489936003360128', '6045489936003364480', '6045489936003364736', '6045490103496970368', '6045490103496970496', '6045490107802132096', '6045490206576177792', '6045490275301101440', '6045490618893154944', '6045490687612596864', '6045490687617964416', '6045490790691916928', '6045490790691917312', '6045490825051658624', '6045490893771170560', '6045501957606694016', '6045501957606730880', '6045501957606731392', '6045501957606738048', '6045501957613991808', '6045501991966471680', '6045501991966472064', '6045501996271737856', '6045502026326261120', '6045502026326266112', '6045502026333452800', '6045502060685987456', '6045502064991291648', '6045502095053034752', '6045502163765297664', '6045502198125052544', '6045502232484715008', '6045502232490018304', '6045502232491819520', '6045502232491836160', '6045502232494258176', '6045502236790001536', '6045502301204217216', '6045502301204248960', '6045502301213734144', '6045502301213737728', '6045502335563963904', '6045502335564002944', '6045502473003049216', '6045502473003060352', '6045502541722480768', '6045502541722509952', '6045502576082225280', '6045502576082238464', '6045502614747130112', '6045502679159938560', '6045502782240663552', '6045503057118542208', '6045503061423822464', '6045503091478302592', '6045503091485234944', '6045503125838013312', '6045503125847460352', '6045503228917299456', '6045503228917314048', '6045503263277056000', '6045503469435509504','6045465639362931584', '6045465639363028608', '6045465643664656256', '6045466189124456448', '6045466193420167296', '6045466193420186368', '6045466193420196608', '6045466193420224768', '6045466193420273920', '6045466193420582528', '6045466326563431424', '6045466326563432448', '6045466326563438464', '6045466330858789376', '6045466330859491072', '6045466365218195328', '6045466429642654720', '6045466429644559360', '6045466433938150528', '6045466463998200960', '6045466464002400512', '6045466468297774720', '6045466468297825024', '6045466498362112896', '6045466502657803264', '6045466532721857024', '6045466532721858048', '6045466567081583488', '6045466571377348864', '6045466571377393920', '6045466571377396864', '6045466605737162624', '6045466640096752640', '6045466640097159296', '6045466640097175040', '6045466670160826496', '6045466670160827648', '6045466670162729344', '6045466704524764160', '6045466708816308992', '6045466708816661888', '6045466738874738944', '6045466743176092672', '6045476565759710592', '6045476600119453440', '6045476634479183488', '6045476634479188480', '6045476634486343168', '6045476634489098880', '6045476638782998528', '6045477252955831552', '6045477356033681152', '6045477394697352192', '6045477459113058048', '6045477532136361856', '6045477630911716992', '6045477635215605888', '6045477944452703488', '6045477978812648576', '6045478043228497024', '6045478043235649408', '6045478047532096640', '6045478077588248832', '6045478077588256000', '6045478077595420288', '6045478180667514240', '6045478184971102208', '6045478184971231360', '6045478215027308544', '6045478283746872064', '6045478318107741440', '6045478318113833344', '6045478322409875328', '6045478356769586304', '6045478356769607936', '6045478356769617280', '6045478391129551488', '6045478391129564160', '6045478391129564544', '6045478425489317760', '6045478489905360384', '6045478524265081344', '6045478524270536320', '6045478631647842688', '6045478696070882304', '6045478700367408256', '6045478730423545984', '6045479005301398400', '6045479280179418624', '6045479451983450240', '6045479525001272448', '6045482887958839040', '6045489038344767360', '6045489077009761024', '6045489107064264576', '6045489141424041600', '6045489278872986880', '6045489656820248320', '6045490137856618624', '6045490210881403904', '6045490240935944192', '6045490515813875200', '6045490550173615744', '6045490687612631936', '6045501957616360448', '6045502030631545856', '6045502060686019840', '6045502369923755776', '6045502511667851264', '6045503263277059584']]


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


