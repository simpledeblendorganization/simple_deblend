### This code will read in the astrobase periodograms to look for 
### the highest peaks and the respective signal-to-noise ratio
### and save those values, and also plot them up

import matplotlib.pyplot as plt
import glob
import pickle
from scipy.stats import sigmaclip
import numpy as np
import heapq
import pickle

from astropy.stats import median_absolute_deviation as MAD

from multiprocessing import Pool
nworkers = 30

#num_periodogram_values_to_check_each_side = 15
median_filter_half_bin_size = 80#40#20
median_filter_half_bin_size_bls = 400#75
nbestpeaks = 12
periodepsilon = .05
freq_window_epsilon = 3.



def parallel_run(input_things):

    ID, ap = input_things

    print(ID)

    ls_values = []
    pdm_values = []
    bls_values = []

    try:
        lsp_dict = pickle.load(open("../../period_search/generalvar_output/" + ID + "_" + id_to_ap[ID] + "_varoutput.pkl","rb"))
    except IOError:
        lsp_dict = pickle.load(open("../../period_search/generalvar_output/" + ID + "_" + id_to_ap[ID] + "_ideal_varoutput.pkl","rb"))

    lc = pickle.load(open("../../tfa/pickled_tfa_output_lc_final_justapused/" + ID + "_" + id_to_ap[ID] + "_pickled_output.p","rb"),encoding='latin')

    for index, results_sav in [(0,ls_values),(1,pdm_values),(2,bls_values)]:
        # Get right half bin size
        if index == 2:
            h = median_filter_half_bin_size_bls
        else:
            h = median_filter_half_bin_size

        # Figure out some median filter stuff
        duration = lc[0][-1] - lc[0][0]
        freq_window_size = freq_window_epsilon/duration
        freq_window_index_size = int(freq_window_size/(1./lsp_dict[index]['periods'][0] - 1./lsp_dict[index]['periods'][1]))
        #freq_window_index_center = freq_window_index_size + h/2

        # Make the median filter
        median_filter = []
        for i in range(len(lsp_dict[index]['lspvals'])):
            temp = []
            if i > freq_window_index_size:
                if i-freq_window_index_size <= 0:
                    raise RuntimeError("Too small, " + str(i-freq_window_index_size))
                temp.extend(lsp_dict[index]['lspvals'][max(0,i-freq_window_index_size-h):i-freq_window_index_size].tolist())
            if i + freq_window_index_size < len(lsp_dict[index]['lspvals']):
                temp.extend(lsp_dict[index]['lspvals'][i+freq_window_index_size:i+freq_window_index_size+h].tolist())
                
            temp = np.array(temp)
            wherefinite = np.isfinite(temp)
            vals, low, upp = sigmaclip(temp[wherefinite],low=3,high=3)
            #print(low,upp)
            median_filter.append(np.median(vals))

        corrected_lspvals = lsp_dict[index]['lspvals'] - median_filter

        #if index == 2:
        #    plt.plot(lsp_dict[index]['periods'],corrected_lspvals,lw=.5)
        #    plt.xscale('log')
        #    plt.savefig("temp.pdf")
        #    raise RuntimeError("just a normal quit")

        # Find 10 maxima, from https://github.com/waqasbhatti/astrobase/blob/master/astrobase/periodbase/zgls.py
        #finitepeakind = np.isfinite(corrected_lspvals)
        #finlsp = corrected_lspvals[finitepeakind]
        #finperiods = periods[finitepeakind]

        if index == 1: # If PDM
            sortedlspind = np.argsort(corrected_lspvals)
        else:
            sortedlspind = np.argsort(corrected_lspvals)[::-1]
        sortedlspperiods = lsp_dict[index]['periods'][sortedlspind]
        sorted_clspvals = corrected_lspvals[sortedlspind]


        ## now get the nbestpeaks
        nbestperiods, nbestlspvals, nbestindices, peakcount = (
            [sortedlspperiods[0]],
            [sorted_clspvals[0]],
            [sortedlspind[0]],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval, i in zip(sortedlspperiods, sorted_clspvals, sortedlspind):

            if peakcount == nbestpeaks:
                break
            perioddiff = abs(period - prevperiod)
            bestperiodsdiff = [abs(period - x) for x in nbestperiods]

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different peak
            # in the periodogram
            if (perioddiff > (periodepsilon*prevperiod) and
                all(x > (periodepsilon*period) for x in bestperiodsdiff)):
                nbestperiods.append(period)
                nbestlspvals.append(lspval)
                nbestindices.append(i)
                peakcount = peakcount + 1

            prevperiod = period

        """
        # Get 10 maxima
        best_ones = heapq.nlargest(10,enumerate(corrected_lspvals),key=lambda o:o[1])#zip(corrected_lspvals,lsp_dict[index]['periods']),key=lambda o: o[0])

        nbestindices = [item[0] for item in best_ones]
        """
        
        del i

        for peak_i in nbestindices:
            #print("           peak num: " + str(peak_i) + ",  value: " + str(lsp_dict[index]['periods'][peak_i]))
            #if peak[0] < num_periodogram_values_to_check_each_side:
            temp = []
            if peak_i > freq_window_index_size:
                temp.extend(lsp_dict[index]['lspvals'][max(0,peak_i-freq_window_index_size-h):peak_i-freq_window_index_size].tolist())
            if peak_i + freq_window_index_size < len(lsp_dict[index]['lspvals']):
                temp.extend(lsp_dict[index]['lspvals'][peak_i+freq_window_index_size:peak_i+freq_window_index_size+h].tolist())
            temp = np.array(temp)
            wherefinite = np.isfinite(temp)
            vals, low, upp = sigmaclip(temp[wherefinite],low=3,high=3)
            stddev = np.std(vals)

            if index == 1: # If PDM
                results_sav.append((lsp_dict[index]['periods'][peak_i],(1.-lsp_dict[index]['lspvals'][peak_i])/stddev))
            else:
                results_sav.append((lsp_dict[index]['periods'][peak_i],lsp_dict[index]['lspvals'][peak_i]/stddev))
         




    return [ID,ls_values,pdm_values,bls_values]




if __name__ == '__main__':


    apused_files = glob.glob("../../tfa/pickled_tfa_output_lc_final_justapused/*.p")


    # Getting the right ap number
    id_to_ap = {}
    for f in apused_files:
        ID = f.split("/")[-1].split("_")[0]
        ap = f.split("/")[-1].split("_")[1]
        id_to_ap[ID] = ap


    pool = Pool(nworkers)
    tasks = [(ID,id_to_ap[ID]) for ID in id_to_ap.keys()]
    print(len(tasks))

    results = pool.map(parallel_run,tasks)

    pool.close()
    pool.join()

    to_dump = {'LS':{},'PDM':{},'BLS':{}}

    for result in results:
        ID = result[0]
        to_dump['LS'][ID] = result[1]
        to_dump['PDM'][ID] = result[2]
        to_dump['BLS'][ID] = result[3]




    pickle.dump(to_dump,
                open("lsp_peak_snr_values_periods.pkl","wb"))
