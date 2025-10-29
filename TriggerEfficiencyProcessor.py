#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept  27 2023

@author: Marcus Converse
Purpose: To read in the processed ALPACA files produced in the first part of the DoAnalysis.sh script and produce the results:
ALPACA outputs are found in /pscratch/sd/m/mconver2/TriggerEfficiencyExtTrigger/$AcqDet/$AcqDet.part$i.root. Save these in home/$AcqDet/WSRunsFolder/GoodFormat/
Make Data Catalog Queries for each of these split lists of runs.
Save the DC Queries output to my cori home directory home/$AcqDet/DCQueries/
"""
import uproot as up
import matplotlib.pyplot as plt
from mpmath import mp
import numpy as np
import matplotlib
import pandas as pd
import os
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
from scipy.interpolate import UnivariateSpline
from pylab import rcParams
rcParams['figure.figsize'] = 13,7 
plt.rcParams.update({'font.size': 14})
import tqdm
from scipy import stats
import pickle
import numba as nb
import time
import scipy
from scipy import optimize
from scipy.interpolate import interp1d
#Load in necessary environment variables


acq_details = os.environ["AcqDet"] 
scratchDirectory = os.environ["SCRATCH"]
top_folder = scratchDirectory+'/TriggerEfficiencyExtTrigger/'+acq_details+'/'
file_paths = [top_folder + x for x in os.listdir(top_folder)]
Binwidth = int(os.environ["binwidth"]) 
upperPAlim = int(os.environ["upperpulsearealimit"])
lowerPAlim = int(os.environ["lowerpulsearealimit"])
dcDictPath = homeDirectory+'/TriggerResults/'+acq_details+'/Results/DC_Dict.npy'
outputpath = homeDirectory+'/TriggerResults/'+acq_details+'/Results/'+acq_details#path for the start of the output path
innerradius0 = int(os.environ["innerradius0"])
innerradius1 = int(os.environ["innerradius1"])
innerradius2 = int(os.environ["innerradius2"])
outerradius0 = int(os.environ["outerradius0"])
outerradius1 = int(os.environ["outerradius1"])
outerradius2 = int(os.environ["outerradius2"])
NOMINAL_THRESHOLD = int(os.environ["SetNominalThreshold"])
#Is the data from WS random triggers?
WIMPSEARCHDATA = os.environ["WIMPSEARCHDATA"]
if WIMPSEARCHDATA == 'True':
    WIMPSearchData = True
else:
    WIMPSearchData = False
#Putting all the binning stuff here so I don't have to hunt for it

#bins for histogramming
#dev_bins = [    0.,    25.,    50.,    75.,   100.,   125.,   150.,   250., 400., 600.,   950.,  1525.,  2400.,  3775.,  5975.,  9475., 15025., 24000.] #Bins that Dev specified. Linear with 25 phd width until 250, then logarithmic up to 24k

# New standard for bins. 20 phd bins from 0 to 500 phd
dev_bins = np.arange(0,501,20) #Changed from 20 to 50 in bin widht and 500 to 1000 for SR1 comparison 
new_bin_centers = np.arange(10,499,20)
new_bin_width = np.ones_like(new_bin_centers)*10

#bins for fitting
#new_bin_centers = [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 200.0, 325.0, 500.0, 775.0, 1237.5, 1962.5, 3087.5, 4875.0, 7725.0, 12250.0, 19400.0]


#new_bin_width = [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 50.0, 75.0, 100.0, 175.0, 287.5, 437.5, 687.5, 1100.0, 1750.0, 2775.0, 4375.0]


#Define fitting range in bins
#We pick this fit range to capture the turn on point and saturation
topbin = 15 #15 bin starting at 300 ending at 320 phd
botbin = 4 # bin starting at 60 ending at 80 phd

#Bins for plotting
#new_bins = [0, 25, 50, 75, 100, 125, 150, 250, 400, 600, 950, 1525, 2400, 3775, 5975, 9475, 15025]

new_bins = dev_bins[:-1] #why did I cut off the last bin?


#Grab alpaca output from scratch and read it into a dataframe

#Grab alpaca output from scratch and read it into a dataframe

n_totrandomtrigs = 0
n_totgpstrigs = 0

n_totevents = 0
n_totevents_preareacut = 0

file1 = up.open(file_paths[0])
datadict1 = {}
for key in file1.keys():
    if 'Metadata' not in key:
        for subkey in file1[key].keys():
            datadict1[subkey] = file1[key][subkey].__array__()
    
masterdf = pd.DataFrame(datadict1)

n_totrandomtrigs = n_totrandomtrigs + max(masterdf['n_totrandomtrigs'].tolist())
n_totgpstrigs = n_totgpstrigs + max(masterdf['n_totgpstrigs'].tolist())

n_totevents = n_totevents + max(masterdf['n_totevents'].tolist())
n_totevents_preareacut = n_totevents_preareacut + max(masterdf['n_totevents_preareacut'].tolist())


for subsequent_file in file_paths[1:]:
    file2 = up.open(subsequent_file)
    datadict2 = {}
    for key in file2.keys():
        if 'Metadata' not in key:
            for subkey in file2[key].keys():
                datadict2[subkey] = file2[key][subkey].__array__()
        df2 = pd.DataFrame(datadict2)
    
    try:
        n_totrandomtrigs = n_totrandomtrigs + max(df2['n_totrandomtrigs'].tolist())
        n_totgpstrigs = n_totgpstrigs + max(df2['n_totgpstrigs'].tolist())

        n_totevents = n_totevents + max(df2['n_totevents'].tolist())
        n_totevents_preareacut = n_totevents_preareacut + max(df2['n_totevents_preareacut'].tolist())
    except:
        print('Something funky with' + subsequent_file)
    
    
    newdf = pd.concat([masterdf,df2],ignore_index=True)
    masterdf = newdf    
    
    

#Cut the master dataframe down based on data quality cuts
tdiffcut = 12500 #6n + coincidence window. 12500 for SR3, 8660 for SR2/SR1
maxpctcut = 40 #max channel area cut
negAreaFracCut = .2 #negative area fraction cut
binwidth = 10 #why did I do this?
maxpa = 24000 #where is this used?
masterdf['MaxChPCT'] = masterdf['MaxChArea']/masterdf['Pulse_Area'] * 100 
masterdf['negAreaFrac'] = np.abs(masterdf['negativeArea_phd'])/masterdf['Pulse_Area']
masterdf['promptFrac200ns'] = masterdf['pulseArea200ns_phd']/masterdf['Pulse_Area']
masterdf['r'] = np.sqrt(masterdf['s2X_cm']**2 + masterdf['s2Y_cm']**2)
masterdf['PL90_10'] = (masterdf['aft90_ns'] - masterdf['aft10_ns'])/1000

masterdf = masterdf[masterdf['rid']>6940] #holdover from SR1 from where 6940 is the last run without the DSM pulse digitized


#The last parameter is Pulse start time is greater than 0, meaning after the trigger. This must be removed for the random trigger only runs.

if WIMPSearchData == True:
    passcutdf = masterdf[masterdf['MaxChPCT']<maxpctcut][masterdf['negAreaFrac']<negAreaFracCut][masterdf['s2XYChi2']>0][masterdf['tdiff_next']>tdiffcut][masterdf['tdiff_prev']>tdiffcut][(masterdf.Pulse_Area< 250 - 245 * masterdf.promptFrac200ns)|(masterdf.promptFrac200ns<0.15)][masterdf['PST']>0]
else:
    passcutdf = masterdf[masterdf['MaxChPCT']<maxpctcut][masterdf['negAreaFrac']<negAreaFracCut][masterdf['s2XYChi2']>0][masterdf['tdiff_next']>tdiffcut][masterdf['tdiff_prev']>tdiffcut][(masterdf.Pulse_Area< 250 - 245 * masterdf.promptFrac200ns)|(masterdf.promptFrac200ns<0.15)] #[masterdf['s2XYChi2']>np.power(10.0,-0.2)][masterdf['s2XYChi2']<np.power(10.0,0.35)] newest DQ cuts...


passcutdf.to_pickle('$HOME/my_dataframe.pkl')

#Histogramming trg and total counts and binning stuff

trgcts,trgbins = np.histogram(passcutdf[passcutdf['trgStatus']==1].Pulse_Area,bins = dev_bins)

totcts,totbins = np.histogram(passcutdf.Pulse_Area,bins = dev_bins)
np.savetxt(outputpath+'trgcts.csv',trgcts)
np.savetxt(outputpath+'totcts.csv',totcts)
np.savetxt(outputpath+'totbins.csv',totbins)

trgcts,trgbins = np.histogram(passcutdf[passcutdf['trgStatus']==1].Pulse_Area,bins = dev_bins)

totcts,totbins = np.histogram(passcutdf.Pulse_Area,bins = dev_bins)

#Slicing in r

innertrgcts,innertrgbins = np.histogram(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0][passcutdf['trgStatus']==1].Pulse_Area,
                              bins = dev_bins)

innertotcts,innertotbins = np.histogram(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0].Pulse_Area,
                              bins = dev_bins)

middletrgcts,middletrgbins = np.histogram(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1][passcutdf['trgStatus']==1].Pulse_Area,
                              bins = dev_bins)

middletotcts,middletotbins = np.histogram(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1].Pulse_Area,
                              bins = dev_bins)

outertrgcts,outertrgbins = np.histogram(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2][passcutdf['trgStatus']==1].Pulse_Area,
                              bins = dev_bins)

outertotcts,outertotbins = np.histogram(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2].Pulse_Area,
                              bins = dev_bins)

plshort = 2.5
pllong = 5
#Slicing in pulse length 90-10
shorttrgcts,shorttrgbins = np.histogram(passcutdf[passcutdf['PL90_10']>0][passcutdf['PL90_10']<plshort][passcutdf['trgStatus']==1].Pulse_Area,
                              bins = dev_bins)

shorttotcts,shorttotbins = np.histogram(passcutdf[passcutdf['PL90_10']>0][passcutdf['PL90_10']<plshort].Pulse_Area,
                              bins = dev_bins)

mediumtrgcts,mediumtrgbins = np.histogram(passcutdf[passcutdf['PL90_10']>plshort][passcutdf['PL90_10']<pllong][passcutdf['trgStatus']==1].Pulse_Area,
                              bins = dev_bins)

mediumtotcts,mediumtotbins = np.histogram(passcutdf[passcutdf['PL90_10']>plshort][passcutdf['PL90_10']<pllong].Pulse_Area,
                              bins = dev_bins)

longtrgcts,longtrgbins = np.histogram(passcutdf[passcutdf['PL90_10']>pllong][passcutdf['trgStatus']==1].Pulse_Area,
                              bins = dev_bins)

longtotcts,longtotbins = np.histogram(passcutdf[passcutdf['PL90_10']>pllong].Pulse_Area,
                              bins = dev_bins)


inner_dnt = max(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0][passcutdf['trgStatus']==0].Pulse_Area)
inner_dnt_noe = len(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0])
inner_dnt_noe_lt120 = len(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0][passcutdf['Pulse_Area']<120])
inner_dnt_noe_gt120 = len(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0][passcutdf['Pulse_Area']>120])
inner_dnt_noe_lt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0][passcutdf['Pulse_Area']<NOMINAL_THRESHOLD])
inner_dnt_noe_gt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['r']>innerradius0][passcutdf['r']<outerradius0][passcutdf['Pulse_Area']>NOMINAL_THRESHOLD])

middle_dnt = max(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1][passcutdf['trgStatus']==0].Pulse_Area)
middle_dnt_noe = len(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1])
middle_dnt_noe_lt120 = len(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1][passcutdf['Pulse_Area']<120])
middle_dnt_noe_gt120 = len(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1][passcutdf['Pulse_Area']>120])
middle_dnt_noe_lt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1][passcutdf['Pulse_Area']<NOMINAL_THRESHOLD])
middle_dnt_noe_gt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['r']>innerradius1][passcutdf['r']<outerradius1][passcutdf['Pulse_Area']>NOMINAL_THRESHOLD])

outer_dnt = max(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2][passcutdf['trgStatus']==0].Pulse_Area)
outer_dnt_noe = len(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2])
outer_dnt_noe_lt120 = len(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2][passcutdf['Pulse_Area']<120])
outer_dnt_noe_gt120 = len(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2][passcutdf['Pulse_Area']>120])
outer_dnt_noe_lt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2][passcutdf['Pulse_Area']<NOMINAL_THRESHOLD])
outer_dnt_noe_gt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['r']>innerradius2][passcutdf['r']<outerradius2][passcutdf['Pulse_Area']>NOMINAL_THRESHOLD])

nlt120 = len(passcutdf[passcutdf['Pulse_Area']<120])
ngt120 = len(passcutdf[passcutdf['Pulse_Area']>120])
nlt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['Pulse_Area']<NOMINAL_THRESHOLD])
ngt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['Pulse_Area']>NOMINAL_THRESHOLD])
nntgt_NOMINAL_THRESHOLD = len(passcutdf[passcutdf['Pulse_Area']>NOMINAL_THRESHOLD][passcutdf['trgStatus']==0])
nt_rid = passcutdf[passcutdf['Pulse_Area']>NOMINAL_THRESHOLD][passcutdf['trgStatus']==0].rid.tolist()
nt_eid = passcutdf[passcutdf['Pulse_Area']>NOMINAL_THRESHOLD][passcutdf['trgStatus']==0].eid.tolist()

nt_r = passcutdf[passcutdf['Pulse_Area']>NOMINAL_THRESHOLD][passcutdf['trgStatus']==0].r.tolist()
nt_pa = passcutdf[passcutdf['Pulse_Area']>NOMINAL_THRESHOLD][passcutdf['trgStatus']==0].Pulse_Area.tolist()


#Print some output for easy parsing
print('\n\n\n\n')
print(acq_details + '\n')

print('Runs Included: \n' + str(np.unique(masterdf.rid)) + '\n')

print('The total number of pulses passing cuts in this analysis is ' + str(len(passcutdf)))

print('The total number of pulses between '+ str(innerradius0) +'<r<' + str(outerradius0) +  ' above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(inner_dnt_noe_gt_NOMINAL_THRESHOLD) +' \n')

print('The largest pulse that did not trigger for '+ str(innerradius0) +'<r<' + str(outerradius0) +  ' is: ' +str(inner_dnt) + ' phd \n')

print('The total number of pulses between '+ str(innerradius1) +'<r<' + str(outerradius1) +  ' above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(middle_dnt_noe_gt_NOMINAL_THRESHOLD) +' \n')


print('The largest pulse that did not trigger for '+ str(innerradius1) +'<r<' + str(outerradius1) +  ' is: ' +str(middle_dnt) + ' phd \n')


print('The total number of pulses between '+ str(innerradius2) +'<r<' + str(outerradius2) +  ' above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(outer_dnt_noe_gt_NOMINAL_THRESHOLD) +' \n')


print('The largest pulse that did not trigger for '+ str(innerradius2) +'<r<' + str(outerradius2) +  ' is: ' +str(outer_dnt) + ' phd \n')

print('The total number of pulses above ' + str(NOMINAL_THRESHOLD)  +' phd that did not trigger is is: ' +str(nntgt_NOMINAL_THRESHOLD) +' \n')

if nntgt_NOMINAL_THRESHOLD>0:
    #If there are pulses above nominal threshold, what are their Run/EventIDs?
    print('The Run ID and Event IDs of the not triggered pulses are: \n')
    
    for i in range(0,len(nt_eid)):
                   #print them in the happy mode for copy pasta into the event viewer
                   print(str(int(nt_rid[i])) + ' ' + str(int(nt_eid[i])) + ' r='+str(int(nt_r[i]))+' pulse area = '+str(int(nt_pa[i])))
                   
    

print('\n The total number of pulses above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(ngt_NOMINAL_THRESHOLD) +' \n')


try:
    print('The total number of random triggers included in this measurement is ' + str(n_totrandomtrigs))
    print('The total number of gps triggers included in this measurement is ' + str(n_totgpstrigs))
    print('The total number of events (before total area cuts) from this dataset is ' + str(n_totevents_preareacut))
    print('The total number of events (after total area cuts) from this dataset is ' + str(n_totevents))
except:
    None


#np.save(outputpath+'inefficiency.npy',triginefficiencyvec)

#np.save(outputpath+'inefficiencynoe.npy',triginefficiencynoe)
#np.save(outputpath+'nglt120_NOMINAL_THRESHOLD.npy',[(nlt120,ngt120),(nlt_NOMINAL_THRESHOLD,ngt_NOMINAL_THRESHOLD)])
#np.save(outputpath+'voxeled_nglt120.npy',triginefficiencynoe_nglt120)
#np.save(outputpath+'voxeled_nglt_NOMINAL_THRESHOLD.npy',triginefficiencynoe_nglt_NOMINAL_THRESHOLD)


# Below here is doing the calceff2 calculation, first, functions:

np.random.seed(seed=int(time.time()))

def combinations(n,r):
    numerator = scipy.special.factorial(n)
    denominator = scipy.special.factorial(r)*scipy.special.factorial(n-r)
    return numerator/denominator

def binominal_probability(n,r,p):
    # write a short function to calculate binomial probability.
    # n = number of events
    # r = number of successes
    # p = probability of success for a single event
    # tested against back of the envelope calculations and get_one_trial_random(n,p).
    # RETURNS probability for 1 test.
    val   = mp.mpf(mp.mpf(scipy.special.comb(n,r,exact=True)) * 
                (mp.mpf(str(p))**mp.mpf(str(r)))*(mp.mpf(str(1-p))**mp.mpf(str(n-r))))
    return val

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 /  (2* stddev**2))

def get_one_trial_random(n,p):
    # a short function to random variation in measuring success
    # n = number of events
    # p = probability of success for a single event
    # ex. If we measure 25 (n) events, with each event having 60% (p=0.6) of success,
    # we expect to see 15 successes and 10 failures. How many do we really see due?
    # Tested against binominal_probability(n,r,p).
    # RETURNS number of failures for 1 test.
    trial = np.random.uniform(0,1,size=n)
    fails = np.sum(trial<=p)
    return fails

def round_sig(x, sig=3):
    if x==0:
        return 0
    else:
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

def get_binom_dist(n,r,step=0.1):
    # A function that produces a normalized binominal distribution,
    # where we present the relative probability of measuring r success
    # out of n events. 
    # This function will loop through possible true succeses rates.
    # n = number of total events. Integer.
    # r = successes we are looking for/i.e., what we measure. Integer.
    # step = size of step to make in determining the array of probabilities to test. 
    # We create an array probs_to_test = i/n where we increase i by step from 0 to n. 
    probs_to_test = [i/n for i in np.arange(0,n+1*step,step)]
    binom_dist = [binominal_probability(n,r,i) 
                  for i in probs_to_test]
    binom_dist_sum = np.sum(binom_dist)
    binom_dist = [x/binom_dist_sum for x in binom_dist]
    return binom_dist,probs_to_test

def get_bounds_median(vals,edges,level):
    # Short function to calcuate the confidence interval based 
    # on a probability disitrbution. Creates a 1d sline to calcualte. 
    # Critically, make the assumption that edges is already finely binned.
    # vals = probability (y)
    # edges = success rates (x)
    # level = confidence interval to use. 1 sigma =0.66, 2 sigma =0.95
    # RETURNS the measured lower, median, and upper bounds.
    summed = np.cumsum(vals)
    interped = interp1d(edges,summed)
    x = interped(edges)
    cut_away = (1-level)/2
    median = edges[np.argwhere(x>0.5)[0][0]]
    lower_bound = edges[np.argwhere(x>cut_away)[0][0]]
    upper_bound = edges[np.argwhere(x>1-cut_away)[0][0]]
    return lower_bound,median,upper_bound

def get_bounds_top(vals,edges,level):
    # Short function to calcuate the confidence interval based 
    # on a probability disitrbution. Creates a 1d sline to calcualte. 
    # Critically, make the assumption that edges is already finely binned.
    # vals = probability (y)
    # edges = success rates (x)
    # level = confidence interval to use. 1 sigma =0.66, 2 sigma =0.95
    # RETURNS the measured lower, median, and upper bounds.
    # UPPER BOUND WILL ALWAYS BE 1.
    summed = np.cumsum(vals)
    interped = interp1d(edges,summed)
    x = interped(edges)
    cut_away = (1-level)
    median = edges[np.argwhere(x>0.5)[0][0]]
    lower_bound = edges[np.argwhere(x>cut_away)[0][0]]
    upper_bound = 1
    return lower_bound,median,upper_bound


def calculate_one_point(n,r,step=0.1,level=0.95):
    binom_outcomes,true_successes = get_binom_dist(n,r,step)
    if r<n:
        bound = get_bounds_median(binom_outcomes,true_successes,level)
    elif r>n:
        print("NOT PHYSICAL")
        return
    else:
        bound = get_bounds_top(binom_outcomes,true_successes,level)
    return bound

def binomial_error(k,n):
    #does binomial errors on efficiency given k triggers and n total events
    #for bins with exceptionally high stats
    if n!=0:
        return np.sqrt(k*(1-k/n))/n
    else:
        return np.nan


##### Now, doing things
#uses trgcts and totcts

eff = []#[trgcts[0]/totcts[0],trgcts[1]/totcts[1],trgcts[2]/totcts[2],trgcts[3]/totcts[3]]
ub = []#[trgcts[0]/totcts[0] + 0.0001,trgcts[1]/totcts[1] + 0.0001,trgcts[2]/totcts[2] + 0.0001,trgcts[3]/totcts[3] + 0.0001]
lb = []#[trgcts[0]/totcts[0] - 0.0001,trgcts[1]/totcts[1] - 0.0001,trgcts[2]/totcts[2] - 0.0001,trgcts[3]/totcts[3] - 0.0001]
one_sigma = []
one_sigma_ub = []
one_sigma_lb = []
high_stats_cutoff = 200
for i in range(0,len(totcts)):
    #for every bin after the 4th grab the total counts and the triggered counts
    temp_totct = totcts[i] 
    temp_trgct = trgcts[i]
    if temp_totct > high_stats_cutoff: #If the bin has really high stats
        temp_eff = temp_trgct/temp_totct #get the effeciency
        temp_ub = temp_eff + binomial_error(temp_trgct,temp_totct)*2 # get the 2-sigma (95% CI) upper bound
        temp_lb = temp_eff - binomial_error(temp_trgct,temp_totct)*2 # get the 2-sigma (95% CI) lower bound
        temp_one_sigma = binomial_error(temp_trgct,temp_totct) #Get the 1-sigma width
        temp_one_sigma_ub = temp_eff + binomial_error(temp_trgct,temp_totct)
        temp_one_sigma_lb = temp_eff - binomial_error(temp_trgct,temp_totct)
    else:    
        #print('Bin : ' + str(i))
        #print('Triggers: ' + str(temp_trgct))
        #print('Total: ' + str(temp_totct))

        if temp_totct == 0:
            temp_eff = np.nan
            #put these as not nan because that screwed up my plot for some reason?
            temp_ub = -1
            temp_lb = -10
            temp_one_sigma = 100
            temp_one_sigma_ub = 100
            temp_one_sigma_lb = -100
        else:
            #grab efficiency
            temp_eff = temp_trgct/temp_totct
            #calculate errors
            try:
                test = calculate_one_point(float(temp_totct),float(temp_trgct))
                temp_lb = test[0]
                temp_ub = test[2]
                one_sig_holder = calculate_one_point(float(temp_totct),float(temp_trgct),level = 0.68)
                temp_one_sigma_ub = one_sig_holder[2]
                temp_one_sigma_lb = one_sig_holder[0]
                #temp_one_sigma = one_sig_holder[2] - one_sig_holder[1] This is how it was on 7/17/24 and it is, I believe, wrong :(
                temp_one_sigma = (one_sig_holder[2] - one_sig_holder[0])/2 # this should be right
            except Exception as error: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
                print('We had an error : '+ type(error).__name__, "–", error)
                temp_lb = temp_eff-0.001
                temp_ub = temp_eff+0.001
                temp_one_sigma = 0.001
                temp_one_sigma_ub = temp_eff +0.001
                temp_one_sigma_lb = temp_eff -0.001
        #store efficiency and errors
        #print('Efficiency: ' + str(temp_eff))
        #print('Lower Bound: ' + str(temp_lb))
        #print('Upper Bound: ' + str(temp_ub))
    eff = eff + [temp_eff]
    ub = ub + [temp_ub]
    lb = lb +[temp_lb]
    one_sigma = one_sigma + [temp_one_sigma]
    one_sigma_ub = one_sigma_ub + [temp_one_sigma_ub]
    one_sigma_lb = one_sigma_lb + [temp_one_sigma_lb]
    
#save efficiency and errors
np.save(outputpath+'efficiency.npy',eff)
np.save(outputpath+'upperbound.npy',ub)
np.save(outputpath+'lowerbound.npy',lb)
np.save(outputpath+'one_sigma.npy',one_sigma)


np.savetxt(outputpath+'efficiency.csv',eff)
np.savetxt(outputpath+'upperbound.csv',ub)
np.savetxt(outputpath+'lowerbound.csv',lb)
np.savetxt(outputpath+'one_sigma.csv',one_sigma)
np.savetxt(outputpath+'one_sigma_lb.csv',one_sigma_lb)
np.savetxt(outputpath+'one_sigma_ub.csv',one_sigma_ub)


#For radius slices

# innereff = []#[innertrgcts[0]/innertotcts[0],innertrgcts[1]/innertotcts[1],innertrgcts[2]/innertotcts[2],innertrgcts[3]/innertotcts[3]]
# innerub = []#[innertrgcts[0]/innertotcts[0] + 0.0001,innertrgcts[1]/innertotcts[1] + 0.0001,innertrgcts[2]/innertotcts[2] + 0.0001,innertrgcts[3]/innertotcts[3] + 0.0001]
# innerlb = []#[innertrgcts[0]/innertotcts[0] - 0.0001,innertrgcts[1]/innertotcts[1] - 0.0001,innertrgcts[2]/innertotcts[2] - 0.0001,innertrgcts[3]/innertotcts[3] - 0.0001]
# for i in range(0,len(innertotcts)):
#     #for every bin after the 4th grab the total counts and the triggered counts 
#     innertemp_totct = innertotcts[i] 
#     innertemp_trgct = innertrgcts[i]
#     #print('Bin : ' + str(i))
#     #print('Triggers: ' + str(temp_trgct))
#     #print('Total: ' + str(temp_totct))
#     if temp_totct > high_stats_cutoff: #If the bin has really high stats
#         innertemp_eff = innertemp_trgct/innertemp_totct #get the effeciency
#         innertemp_ub = innertemp_eff + binomial_error(innertemp_trgct,innertemp_totct)*2 # get the 2-sigma (95% CI) upper bound
#         innertemp_lb = innertemp_eff + binomial_error(innertemp_trgct,innertemp_totct)*2 # get the 2-sigma (95% CI) lower bound
#     else:    

#         if innertemp_totct == 0:
#             innertemp_eff = np.nan
#             #put these as not nan because that screwed up my plot for some reason?
#             innertemp_ub = -1
#             innertemp_lb = -10
#         else:
#             #grab efficiency
#             innertemp_eff = innertemp_trgct/innertemp_totct
#             #calculate errors
#             try:
#                 test = calculate_one_point(float(innertemp_totct),float(innertemp_trgct))
#                 innertemp_lb = test[0]
#                 innertemp_ub = test[2]
#             except: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
#                 innertemp_lb = innertemp_eff-0.001
#                 innertemp_ub = innertemp_eff+0.001

#     innereff = innereff + [innertemp_eff]
#     innerub = innerub + [innertemp_ub]
#     innerlb = innerlb +[innertemp_lb]


# middleeff = []#[middletrgcts[0]/middletotcts[0],middletrgcts[1]/middletotcts[1],middletrgcts[2]/middletotcts[2],middletrgcts[3]/middletotcts[3]]
# middleub = []#[middletrgcts[0]/middletotcts[0] + 0.0001,middletrgcts[1]/middletotcts[1] + 0.0001,middletrgcts[2]/middletotcts[2] + 0.0001,middletrgcts[3]/middletotcts[3] + 0.0001]
# middlelb = []#[middletrgcts[0]/middletotcts[0] - 0.0001,middletrgcts[1]/middletotcts[1] - 0.0001,middletrgcts[2]/middletotcts[2] - 0.0001,middletrgcts[3]/middletotcts[3] - 0.0001]
# for i in range(0,len(middletotcts)):
#     #for every bin after the 4th grab the total counts and the triggered counts 
#     middletemp_totct = middletotcts[i] 
#     middletemp_trgct = middletrgcts[i]
#     #print('Bin : ' + str(i))
#     #print('Triggers: ' + str(temp_trgct))
#     #print('Total: ' + str(temp_totct))
#     if temp_totct > high_stats_cutoff: #If the bin has really high stats
#         middletemp_eff = middletemp_trgct/middletemp_totct #get the effeciency
#         middletemp_ub = middletemp_eff + binomial_error(middletemp_trgct,middletemp_totct)*2 # get the 2-sigma (95% CI) upper bound
#         middletemp_lb = middletemp_eff + binomial_error(middletemp_trgct,middletemp_totct)*2 # get the 2-sigma (95% CI) lower bound
#     else:    

#         if middletemp_totct == 0:
#             middletemp_eff = np.nan
#             #put these as not nan because that screwed up my plot for some reason?
#             middletemp_ub = -1
#             middletemp_lb = -10
#         else:
#             #grab efficiency
#             middletemp_eff = middletemp_trgct/middletemp_totct
#             #calculate errors
#             try:
#                 test = calculate_one_point(float(middletemp_totct),float(middletemp_trgct))
#                 middletemp_lb = test[0]
#                 middletemp_ub = test[2]
#             except: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
#                 middletemp_lb = middletemp_eff-0.001
#                 middletemp_ub = middletemp_eff+0.001

#     middleeff = middleeff + [middletemp_eff]
#     middleub = middleub + [middletemp_ub]
#     middlelb = middlelb +[middletemp_lb]

# outereff = []#[outertrgcts[0]/outertotcts[0],outertrgcts[1]/outertotcts[1],outertrgcts[2]/outertotcts[2],outertrgcts[3]/outertotcts[3]]
# outerub = []#[outertrgcts[0]/outertotcts[0] + 0.0001,outertrgcts[1]/outertotcts[1] + 0.0001,outertrgcts[2]/outertotcts[2] + 0.0001,outertrgcts[3]/outertotcts[3] + 0.0001]
# outerlb = []#[outertrgcts[0]/outertotcts[0] - 0.0001,outertrgcts[1]/outertotcts[1] - 0.0001,outertrgcts[2]/outertotcts[2] - 0.0001,outertrgcts[3]/outertotcts[3] - 0.0001]
# for i in range(0,len(outertotcts)):
#     #for every bin after the 4th grab the total counts and the triggered counts 
#     outertemp_totct = outertotcts[i] 
#     outertemp_trgct = outertrgcts[i]
#     #print('Bin : ' + str(i))
#     #print('Triggers: ' + str(temp_trgct))
#     #print('Total: ' + str(temp_totct))
#     if temp_totct > high_stats_cutoff: #If the bin has really high stats
#         outertemp_eff = outertemp_trgct/outertemp_totct #get the effeciency
#         outertemp_ub = outertemp_eff + binomial_error(outertemp_trgct,outertemp_totct)*2 # get the 2-sigma (95% CI) upper bound
#         outertemp_lb = outertemp_eff + binomial_error(outertemp_trgct,outertemp_totct)*2 # get the 2-sigma (95% CI) lower bound
#     else:    

#         if outertemp_totct == 0:
#             outertemp_eff = np.nan
#             #put these as not nan because that screwed up my plot for some reason?
#             outertemp_ub = -1
#             outertemp_lb = -10
#         else:
#             #grab efficiency
#             outertemp_eff = outertemp_trgct/outertemp_totct
#             #calculate errors
#             try:
#                 test = calculate_one_point(float(outertemp_totct),float(outertemp_trgct))
#                 outertemp_lb = test[0]
#                 outertemp_ub = test[2]
#             except: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
#                 outertemp_lb = outertemp_eff-0.001
#                 outertemp_ub = outertemp_eff+0.001

#     outereff = outereff + [outertemp_eff]
#     outerub = outerub + [outertemp_ub]
#     outerlb = outerlb +[outertemp_lb]

# # #For PL90-10 slices
   
# shorteff = []#[shorttrgcts[0]/shorttotcts[0],shorttrgcts[1]/shorttotcts[1],shorttrgcts[2]/shorttotcts[2],shorttrgcts[3]/shorttotcts[3]]
# shortub = []#[shorttrgcts[0]/shorttotcts[0] + 0.0001,shorttrgcts[1]/shorttotcts[1] + 0.0001,shorttrgcts[2]/shorttotcts[2] + 0.0001,shorttrgcts[3]/shorttotcts[3] + 0.0001]
# shortlb = []#[shorttrgcts[0]/shorttotcts[0] - 0.0001,shorttrgcts[1]/shorttotcts[1] - 0.0001,shorttrgcts[2]/shorttotcts[2] - 0.0001,shorttrgcts[3]/shorttotcts[3] - 0.0001]
# for i in range(0,len(shorttotcts)):
#     #for every bin after the 4th grab the total counts and the triggered counts 
#     shorttemp_totct = shorttotcts[i] 
#     shorttemp_trgct = shorttrgcts[i]
#     #print('Bin : ' + str(i))
#     #print('Triggers: ' + str(temp_trgct))
#     #print('Total: ' + str(temp_totct))
#     if temp_totct > high_stats_cutoff: #If the bin has really high stats
#         shorttemp_eff = shorttemp_trgct/shorttemp_totct #get the effeciency
#         shorttemp_ub = shorttemp_eff + binomial_error(shorttemp_trgct,shorttemp_totct)*2 # get the 2-sigma (95% CI) upper bound
#         shorttemp_lb = shorttemp_eff + binomial_error(shorttemp_trgct,shorttemp_totct)*2 # get the 2-sigma (95% CI) lower bound
#     else:    

#         if shorttemp_totct == 0:
#             shorttemp_eff = np.nan
#             #put these as not nan because that screwed up my plot for some reason?
#             shorttemp_ub = -1
#             shorttemp_lb = -10
#         else:
#             #grab efficiency
#             shorttemp_eff = shorttemp_trgct/shorttemp_totct
#             #calculate errors
#             try:
#                 test = calculate_one_point(float(shorttemp_totct),float(shorttemp_trgct))
#                 shorttemp_lb = test[0]
#                 shorttemp_ub = test[2]
#             except: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
#                 shorttemp_lb = shorttemp_eff-0.001
#                 shorttemp_ub = shorttemp_eff+0.001

#     shorteff = shorteff + [shorttemp_eff]
#     shortub = shortub + [shorttemp_ub]
#     shortlb = shortlb +[shorttemp_lb]

# mediumeff = []    
# mediumub = []#[mediumtrgcts[0]/mediumtotcts[0] + 0.0001,mediumtrgcts[1]/mediumtotcts[1] + 0.0001,mediumtrgcts[2]/mediumtotcts[2] + 0.0001,mediumtrgcts[3]/mediumtotcts[3] + 0.0001]
# mediumlb = []#[mediumtrgcts[0]/mediumtotcts[0] - 0.0001,mediumtrgcts[1]/mediumtotcts[1] - 0.0001,mediumtrgcts[2]/mediumtotcts[2] - 0.0001,mediumtrgcts[3]/mediumtotcts[3] - 0.0001]
# for i in range(0,len(mediumtotcts)):
#     #for every bin after the 4th grab the total counts and the triggered counts 
#     mediumtemp_totct = mediumtotcts[i] 
#     mediumtemp_trgct = mediumtrgcts[i]
#     #print('Bin : ' + str(i))
#     #print('Triggers: ' + str(temp_trgct))
#     #print('Total: ' + str(temp_totct))
#     if temp_totct > high_stats_cutoff: #If the bin has really high stats
#         mediumtemp_eff = mediumtemp_trgct/mediumtemp_totct #get the effeciency
#         mediumtemp_ub = mediumtemp_eff + binomial_error(mediumtemp_trgct,mediumtemp_totct)*2 # get the 2-sigma (95% CI) upper bound
#         mediumtemp_lb = mediumtemp_eff + binomial_error(mediumtemp_trgct,mediumtemp_totct)*2 # get the 2-sigma (95% CI) lower bound
#     else:    

#         if mediumtemp_totct == 0:
#             mediumtemp_eff = np.nan
#             #put these as not nan because that screwed up my plot for some reason?
#             mediumtemp_ub = -1
#             mediumtemp_lb = -10
#         else:
#             #grab efficiency
#             mediumtemp_eff = mediumtemp_trgct/mediumtemp_totct
#             #calculate errors
#             try:
#                 test = calculate_one_point(float(mediumtemp_totct),float(mediumtemp_trgct))
#                 mediumtemp_lb = test[0]
#                 mediumtemp_ub = test[2]
#             except: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
#                 mediumtemp_lb = mediumtemp_eff-0.001
#                 mediumtemp_ub = mediumtemp_eff+0.001

#     mediumeff = mediumeff + [mediumtemp_eff]
#     mediumub = mediumub + [mediumtemp_ub]
#     mediumlb = mediumlb +[mediumtemp_lb]    
    
# longeff = []#[longtrgcts[0]/longtotcts[0],longtrgcts[1]/longtotcts[1],longtrgcts[2]/longtotcts[2],longtrgcts[3]/longtotcts[3]]
# longub = []#[longtrgcts[0]/longtotcts[0] + 0.0001,longtrgcts[1]/longtotcts[1] + 0.0001,longtrgcts[2]/longtotcts[2] + 0.0001,longtrgcts[3]/longtotcts[3] + 0.0001]
# longlb = []#[longtrgcts[0]/longtotcts[0] - 0.0001,longtrgcts[1]/longtotcts[1] - 0.0001,longtrgcts[2]/longtotcts[2] - 0.0001,longtrgcts[3]/longtotcts[3] - 0.0001]
# for i in range(0,len(longtotcts)):
#     #for every bin after the 4th grab the total counts and the triggered counts 
#     longtemp_totct = longtotcts[i] 
#     longtemp_trgct = longtrgcts[i]
#     #print('Bin : ' + str(i))
#     #print('Triggers: ' + str(temp_trgct))
#     #print('Total: ' + str(temp_totct))
#     if temp_totct > high_stats_cutoff: #If the bin has really high stats
#         longtemp_eff = longtemp_trgct/longtemp_totct #get the effeciency
#         longtemp_ub = longtemp_eff + binomial_error(longtemp_trgct,longtemp_totct)*2 # get the 2-sigma (95% CI) upper bound
#         longtemp_lb = longtemp_eff + binomial_error(longtemp_trgct,longtemp_totct)*2 # get the 2-sigma (95% CI) lower bound
#     else:    

#         if longtemp_totct == 0:
#             longtemp_eff = np.nan
#             #put these as not nan because that screwed up my plot for some reason?
#             longtemp_ub = -1
#             longtemp_lb = -10
#         else:
#             #grab efficiency
#             longtemp_eff = longtemp_trgct/longtemp_totct
#             #calculate errors
#             try:
#                 test = calculate_one_point(float(longtemp_totct),float(longtemp_trgct))
#                 longtemp_lb = test[0]
#                 longtemp_ub = test[2]
#             except: #Overflow errors assume very high stats. Set as symmetric 0.1% error on efficiency until this can be resolved
#                 longtemp_lb = longtemp_eff-0.001
#                 longtemp_ub = longtemp_eff+0.001

#     longeff = longeff + [longtemp_eff]
#     longub = longub + [longtemp_ub]
#     longlb = longlb +[longtemp_lb]##Fit Sigmoid


#Import Packages and define our fit function
from scipy.optimize import curve_fit
import scipy.interpolate
#Fit function is a sigmoid
def sigmoid(x, x0, k):
    #x0 is the mark where we hit 50% efficiency
    #k is the speed of that change (larger means faster)
    y = 1 / (1 + np.exp(-k*(x-x0))) 
    return (y)
#Initial parameter guesses, pretty good for 1150x6
p0 = [100,0.02]


#Handle NANs in the fit
nan_indices = np.argwhere(np.isnan(eff))
eff = np.delete(eff, nan_indices)

new_bin_centers = np.delete(new_bin_centers, nan_indices)
new_bin_width = np.delete(new_bin_width,nan_indices)
new_bins = np.delete(new_bins,nan_indices)
one_sigma = np.delete(one_sigma,nan_indices)
ub = np.delete(ub,nan_indices)
lb = np.delete(lb,nan_indices)
totcts = np.delete(totcts,nan_indices)

#Perform our fit
popt, pcov = curve_fit(sigmoid,new_bin_centers[botbin:topbin], eff[botbin:topbin],p0, np.array(one_sigma[botbin:topbin]),absolute_sigma = True)
#Generate our best fit curve
x0 = popt[0]
k = popt[1]


fitstep = 0.1
bestfit_belowrange = sigmoid(np.arange(0,new_bin_centers[0],fitstep),x0,k)

bestfit = sigmoid(np.arange(new_bin_centers[0],new_bin_centers[-1],fitstep),x0,k)

x0error = np.sqrt(pcov[0][0])
kerror = np.sqrt(pcov[1][1])
#Find where the 10,50,95% thresholds are in our fit

ten_pct_threshold = x0 + np.log(0.1/(1-0.1))/k #Got this formula from a mathematica solver hashtag quickmaths

ten_pct_threshold_error = np.sqrt(x0error**2  + kerror**2 * (np.log(0.1/(1-0.1))/(k**2))**2) #partial derivatives do work that way sometimes now don't they

fifty_pct_threshold = x0 + np.log(0.5/(1-0.5))/k

fifty_pct_threshold_error = x0error #after simplification of the algebra

ninetyfive_pct_threshold = x0 + np.log(0.95/(1-0.95))/k 

ninetyfive_pct_threshold_error = np.sqrt(x0error**2  + kerror**2 * (np.log(0.95/(1-0.95))/(k**2))**2)

#Print out our fit params and our thresholds. We don't care about the 10% threshold because that's out of our fit range.

print('For the sigmoid fit we get the covariance on x0 to be ' +str(pcov[0][0])+ ' and the covariance on k to be ' +str(pcov[1][1])+ '\n')

print('For the sigmoid fit we get: x0 = ' +str(x0)+ '+/- ' + str(x0error) +' and k = ' +str(k) + '+/- ' +str(kerror))

#Need to add calculating the thresholds with uncertainty on them.

#print('For the sigmoid fit we get the ten percent threshold is ' + str(ten_pct_threshold) + '%')
print('For the sigmoid fit we get the fifty percent threshold is ' + str(fifty_pct_threshold) + '+/-' + str(fifty_pct_threshold_error) + ' phd')
print('For the sigmoid fit we get the ninety-five percent threshold is ' + str(ninetyfive_pct_threshold) + '+/-' + str(ninetyfive_pct_threshold_error) + ' phd')



#Generate the main figure we all know and love
fig,ax = plt.subplots(figsize = (5,5))
#Plot our measured efficiency
ax.errorbar(new_bin_centers,eff,
             xerr=new_bin_width,
         color='red',ls='none',
        marker='o',markersize=5,capsize = 3, label = 'Measured Efficiency')

#Plot our 95% confidence interval
ax.fill_between(new_bins,lb,ub,
                facecolor='blue',alpha=0.25,label='2-Sigma CI',step="post")
#Plot the best fit sigmoid

ax.plot(np.arange(0,new_bin_centers[botbin]-new_bin_width[botbin],fitstep),sigmoid(np.arange(0,new_bin_centers[botbin]-new_bin_width[botbin],fitstep),x0,k),ls = '--',color = 'red')
ax.plot(np.arange(new_bin_centers[botbin]-new_bin_width[botbin],new_bin_centers[-1],fitstep),sigmoid(np.arange(new_bin_centers[botbin]-new_bin_width[botbin],new_bin_centers[-1],fitstep),x0,k), label =acq_details, color = 'red')

#Plot the reference sigmoid from the Pre-SR2 Measurement
#referenceX0 = 103.34912836873175
#referenceK = 0.051729445751319096
#ax.plot(np.arange(0,1000,0.1),sigmoid(np.arange(0,1000,0.1),referenceX0,referenceK), label = 'Reference Fit - Pre SR2', color = 'blue')

#Plot the reference sigmoid from the Pre-SR3 Measurement
referenceX0 = 107.41461916763045  #113.05822844788766 - value from LZAP 5.5.3
referenceK =   0.04865394289861963   #0.04913484487613414 - value from LZAP 5.5.3
fitregion = new_bin_centers[botbin:topbin]

#Plots outside the fit range in dashed line

ax.plot(np.arange(0,new_bin_centers[botbin]-new_bin_width[botbin],fitstep),sigmoid(np.arange(0,new_bin_centers[botbin]-new_bin_width[botbin],fitstep),referenceX0,referenceK),ls='--',color='blue')

ax.plot(np.arange(new_bin_centers[botbin]-new_bin_width[botbin],new_bin_centers[-1],fitstep),sigmoid(np.arange(new_bin_centers[botbin]-new_bin_width[botbin],new_bin_centers[-1],fitstep),referenceX0,referenceK), label = 'Pre SR3 LZAP 5.8.0', color = 'blue')



#Make the plot pretty
#plt.xscale("log")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel("Efficiency",fontsize=15)
plt.xlabel("Pulse Area [phd]",fontsize=15)
plt.axis([1e1,4e2,-0.05,1.05])
ax.legend()
ax.tick_params(axis ='both',which = 'major', direction = 'in', length = 10)
ax.tick_params(axis ='both',which = 'minor', direction = 'in', length = 5)
plt.tight_layout()
#Save the plot
plt.savefig(outputpath+'_eff.png',facecolor = 'white', transparent = False)


fitrange_measured_efficiency = eff[botbin:topbin]
fitrange_bincenters = new_bin_centers[botbin:topbin]
fitrange_one_sigma_error = np.array(one_sigma[botbin:topbin])

fitk = k
fitx0 = x0
fitkerror = kerror
fitx0error = x0error
sigma_n = 20
stepsize_sigma = 1/10
kstep1 = np.ones(int(2*sigma_n/stepsize_sigma)+1) * fitk 
kstep2 = fitkerror * np.arange(-sigma_n,sigma_n +0.0001,stepsize_sigma)
kstep2[int(sigma_n / stepsize_sigma)] = 0
k_array = kstep1+kstep2

x0step1 = np.ones(int(2*sigma_n/stepsize_sigma)+1) * fitx0 
x0step2 = fitx0error * np.arange(-sigma_n,sigma_n +0.0001,stepsize_sigma)
x0step2[int(sigma_n / stepsize_sigma)] = 0
x0_array = x0step1+x0step2

def chi2(data,fit,one_sigma_error):
    return np.sum(((data-fit)**2)/(one_sigma_error))

chi_squared_array = np.zeros([int(2*sigma_n/stepsize_sigma)+1,int(2*sigma_n/stepsize_sigma)+1])
for x in range(0,int(2*sigma_n/stepsize_sigma)+1):
    for y in range(0,int(2*sigma_n/stepsize_sigma)+1):
        chi_squared_array[x,y] = chi2(fitrange_measured_efficiency, sigmoid(fitrange_bincenters,x0_array[y],k_array[x]),fitrange_one_sigma_error)

# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(x0_array, k_array)

z = chi_squared_array
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = 0, 0.6


chi2fig, chi2ax = plt.subplots(figsize = (5,5))

c = chi2ax.pcolormesh(x, y, z, vmin=z_min, vmax=z_max,shading = 'flat')
minx = np.argwhere(chi_squared_array == chi_squared_array.min())[0][0]
miny = np.argwhere(chi_squared_array == chi_squared_array.min())[0][1]
chi2ax.scatter(x[minx,miny],y[minx,miny],marker = 'x',color = 'red',label = '$\chi^2$ minimum')

# set the limits of the plot to the limits of the data
chi2ax.axis([x.min(), x.max(), y.min(), y.max()])
plt.xlabel('k - Best fit = ' + str(round(fitk,4)) + ' +/-' + str(round(fitkerror,4)))
plt.ylabel('x0 - Best fit = ' + str(round(fitx0,1)) + ' +/-' + str(round(fitx0error,1)))

chi2fig.colorbar(c, ax=chi2ax,label = '$\chi^2$')
plt.ylim(90,120)
plt.xlim(0.02,0.1)
plt.tight_layout()
plt.savefig(outputpath+'_chi2map.png',facecolor = 'white',transparent = False)

# #Generate the radius slice figure we all know and love
# rfig,rax = plt.subplots(figsize = (8,7))
# #Plot our measured efficiency
# rax.errorbar(new_bin_centers,innereff,
#              xerr=new_bin_width,
#              yerr = [np.array(innereff)-np.array(innerlb),np.array(innerub)-np.array(innereff)],
#          color='red',ls='none',
#         marker='o',markersize=5,capsize = 3, label = 'R < '+str(outerradius0) + ' cm')
# #rax.fill_between(new_bins,innerlb,innerub,
# #                facecolor='red',alpha=0.25,label='R < '+str(outerradius0) +' cm 2-Sigma CI',step="post")

# rax.errorbar(new_bin_centers,middleeff,
#              xerr=new_bin_width,
#              yerr = [np.array(middleeff)-np.array(middlelb),np.array(middleub)-np.array(middleeff)],
#          color='blue',ls='none',
#         marker='o',markersize=5,capsize = 3, label = str(innerradius1)+' R < '+str(outerradius1) + ' cm')
# #rax.fill_between(new_bins,middlelb,middleub,
# #                facecolor='blue',alpha=0.25,label=str(innerradius1)+' R < '+str(outerradius1) + ' cm 2-Sigma CI',step="post")

# rax.errorbar(new_bin_centers,outereff,
#              xerr=new_bin_width,
#              yerr = [np.array(outereff)-np.array(outerlb),np.array(outerub)-np.array(outereff)],
#          color='green',ls='none',
#         marker='o',markersize=5,capsize = 3, label = str(innerradius2)+' R < '+str(outerradius2) + ' cm')
# #rax.fill_between(new_bins,outerlb,outerub,
# #                facecolor='green',alpha=0.25,label=str(innerradius2)+' R < '+str(outerradius2) + ' cm 2-Sigma CI',step="post")



# #Plot our 95% confidence interval

# #Plot the best fit sigmoid
# rax.plot(np.arange(0,1000,0.1),bestfit, label = 'Full Volume Sigmoid Fit', color = 'red')

# #Make the plot pretty
# #plt.xscale("log")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.ylabel("Efficiency",fontsize=15)
# plt.xlabel("Pulse Area [phd]",fontsize=15)
# plt.axis([1e1,4e2,-0.05,1.05])
# rax.legend()
# rax.tick_params(axis ='both',which = 'major', direction = 'in', length = 10)
# rax.tick_params(axis ='both',which = 'minor', direction = 'in', length = 5)

# #Save the plot
# plt.savefig(outputpath+'_rsliced_eff.png',facecolor = 'white', transparent = False)


# #Generate the Pulse Length slice figure we all know and love
# tfig,tax = plt.subplots(figsize = (8,7))
# #Plot our measured efficiency

# tax.errorbar(new_bin_centers,shorteff,
#              xerr=new_bin_width,
#              yerr = [np.array(shorteff)-np.array(shortlb),np.array(shortub)-np.array(shorteff)],
#          color='red',ls='none',
#         marker='o',markersize=5,capsize = 3, label = 'PL90-10 < '+str(plshort)+' us')
# #tax.fill_between(new_bins,shortlb,shortub,
# #                facecolor='red',alpha=0.25,label='PL90-10 < 3 us 2-Sigma CI',step="post")

# tax.errorbar(new_bin_centers,mediumeff,
#              xerr=new_bin_width,
#              yerr = [np.array(mediumeff)-np.array(mediumlb),np.array(mediumub)-np.array(mediumeff)],
#          color='blue',ls='none',
#         marker='o',markersize=5,capsize = 3, label = str(plshort)+' us < PL90-10 < '+str(pllong)+' us')
# #tax.fill_between(new_bins,mediumlb,mediumub,
# #                facecolor='blue',alpha=0.25,label='3 us < PL90-10 < 6 us 2-Sigma CI',step="post")

# tax.errorbar(new_bin_centers,longeff,
#              xerr=new_bin_width,
#              yerr = [np.array(longeff)-np.array(longlb),np.array(longub)-np.array(longeff)],
#          color='green',ls='none',
#         marker='o',markersize=5,capsize = 3, label = str(pllong)+' us < PL90-10')
# #tax.fill_between(new_bins,longlb,longub,
# #                facecolor='green',alpha=0.25,label='6 us < PL90-10 2-Sigma CI',step="post")



# #Plot our 95% confidence interval

# #Plot the best fit sigmoid
# tax.plot(np.arange(0,1000,0.1),bestfit, label = 'Full Volume Sigmoid Fit', color = 'red')

# #Make the plot pretty
# #plt.xscale("log")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.ylabel("Efficiency",fontsize=15)
# plt.xlabel("Pulse Area [phd]",fontsize=15)
# plt.axis([1e1,4e2,-0.05,1.05])
# tax.legend()
# tax.tick_params(axis ='both',which = 'major', direction = 'in', length = 10)
# tax.tick_params(axis ='both',which = 'minor', direction = 'in', length = 5)

# #Save the plot
# plt.savefig(outputpath+'_tsliced_eff.png',facecolor = 'white', transparent = False)


datadict = {}
datadict['AcquisitionDetails'] = acq_details
datadict['bins'] = new_bins
datadict['bin_centers'] = new_bin_centers
datadict['efficiency'] = eff
datadict['upperbound'] = ub
datadict['lowerbound'] = lb
datadict['confidence_level']=0.95
datadict['one_sigma'] = one_sigma
datadict['one_sigma_lb'] = one_sigma_lb
datadict['one_sigma_ub'] = one_sigma_ub
datadict['x0'] = x0
datadict['x0_error'] = x0error
datadict['k'] = k
datadict['k_error'] = kerror
datadict['95%_Threshold'] = ninetyfive_pct_threshold
datadict['95%_Threshold_error'] = ninetyfive_pct_threshold_error
datadict['fit_bounds'] = [new_bins[botbin],new_bins[topbin]]
datadict['total_counts'] = totcts
datadict['n_gps_triggers'] = n_totgpstrigs
datadict['n_random_triggers'] = n_totrandomtrigs
datadict['n_total_events'] = n_totevents_preareacut
datadict['pulses_over_nominal_threshold'] = ngt_NOMINAL_THRESHOLD

datadict['pulses_over_nominal_threshold_not_triggered'] = inner_dnt_noe_gt_NOMINAL_THRESHOLD + middle_dnt_noe_gt_NOMINAL_THRESHOLD + outer_dnt_noe_gt_NOMINAL_THRESHOLD

datadict['nominal_threshold'] = NOMINAL_THRESHOLD
dc_dict = np.load(dcDictPath,allow_pickle =True)
datadict['DC_info'] = dc_dict
datadict[str(innerradius0) +'<r<' + str(outerradius0) +  '_>' + str(NOMINAL_THRESHOLD)  +'_phd'] = inner_dnt_noe_gt_NOMINAL_THRESHOLD
datadict[str(innerradius0) +'<r<' + str(outerradius0) +  '_largest_not_triggered_pulse_phd'] = inner_dnt

datadict[str(innerradius1) +'<r<' + str(outerradius1) +  '_>' + str(NOMINAL_THRESHOLD)  +'_phd'] = middle_dnt_noe_gt_NOMINAL_THRESHOLD
datadict[str(innerradius1) +'<r<' + str(outerradius1) +  '_largest_not_triggered_pulse_phd'] = middle_dnt

datadict[str(innerradius2) +'<r<' + str(outerradius2) +  '_>' + str(NOMINAL_THRESHOLD)  +'_phd'] = outer_dnt_noe_gt_NOMINAL_THRESHOLD
datadict[str(innerradius2) +'<r<' + str(outerradius2) +  '_largest_not_triggered_pulse_phd'] = outer_dnt



np.save(outputpath+'_resultsdict.npy',datadict)

#Start a txt file to output a report
with open(outputpath + '_report.txt','w') as report:
    report.write('Report on '+acq_details +' measurement \n')
    
    report.write('Runs Included: \n' + str(np.unique(masterdf.rid)) + '\n')
    
    report.write('The total number of pulses passing cuts in this analysis is ' + str(len(passcutdf))+'\n')
    
    report.write('The total number of pulses between '+ str(innerradius0) +'<r<' + str(outerradius0) +  ' above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(inner_dnt_noe_gt_NOMINAL_THRESHOLD) +' \n')
    
    report.write('The largest pulse that did not trigger for '+ str(innerradius0) +'<r<' + str(outerradius0) +  ' is: ' +str(inner_dnt) + ' phd \n')
    
    report.write('The total number of pulses between '+ str(innerradius1) +'<r<' + str(outerradius1) +  ' above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(middle_dnt_noe_gt_NOMINAL_THRESHOLD) +' \n')
    
    report.write('The largest pulse that did not trigger for '+ str(innerradius1) +'<r<' + str(outerradius1) +  ' is: ' +str(middle_dnt) + ' phd \n')
    
    report.write('The total number of pulses between '+ str(innerradius2) +'<r<' + str(outerradius2) +  ' above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(outer_dnt_noe_gt_NOMINAL_THRESHOLD) +' \n')
    
    report.write('The largest pulse that did not trigger for '+ str(innerradius2) +'<r<' + str(outerradius2) +  ' is: ' +str(outer_dnt) + ' phd \n')

    
    
    if nntgt_NOMINAL_THRESHOLD>0:
    #If there are pulses above nominal threshold, what are their Run/EventIDs?
        report.write('The Run ID and Event IDs of the not triggered pulses are: \n')
                     
        for i in range(0,len(nt_eid)):
            report.write(str(int(nt_rid[i])) + ' ' + str(int(nt_eid[i])) + '\n')
                   
    
    
    report.write('For the sigmoid fit we get: x0 = ' +str(x0)+ ' +/- ' + str(x0error) +' and k = ' +str(k) + ' +/- ' +str(kerror) + '\n')
    
    report.write('For the sigmoid fit we get the fifty percent threshold is ' + str(fifty_pct_threshold) + '+/-' + str(fifty_pct_threshold_error) + ' phd\n')
    report.write('For the sigmoid fit we get the ninety-five percent threshold is ' + str(ninetyfive_pct_threshold) + '+/-' + str(ninetyfive_pct_threshold_error) + ' phd\n')  

    
    report.write('\n The total number of pulses above ' + str(NOMINAL_THRESHOLD)  +' phd is: ' +str(ngt_NOMINAL_THRESHOLD) +' \n')
    
    report.write('The total number of random triggers included in this measurement is ' + str(n_totrandomtrigs) + '\n')
    report.write('The total number of gps triggers included in this measurement is ' + str(n_totgpstrigs) + '\n')
    report.write('The total number of events (before total area cuts) from this dataset is ' + str(n_totevents_preareacut) +'\n')
    report.write('The total number of events (after total area cuts) from this dataset is ' + str(n_totevents))
    
    
    report.close()

