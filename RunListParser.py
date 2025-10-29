#!/usr/bin/env pytho3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept  27 2023

@author: Marcus Converse
Purpose: To read in a list of poorly formatted runs and produce:
A properly formatted list of runs split into split_n length run lists parseable by the data catalog. Save these in home/$AcqDet/WSRunsFolder/GoodFormat/
Make Data Catalog Queries for each of these split lists of runs.
Save the DC Queries output to my cori home directory home/$AcqDet/DCQueries/
"""
import os 
import numpy as np

#Load in environment variables.
AcqDet = os.environ["AcqDet"]#acquisition details
homeDirectory = os.environ["WORKDIR"]#homeDirectory
poorformatrunlist_path = os.environ["pfrlp"]#acquisition details
lzapversion = os.environ['lzapversion']
split_n = int(os.environ["split_n"])#number of runs per split
#Load in the full poorly formatted run list.
file = np.loadtxt(poorformatrunlist_path,ndmin=1)

#Define the function to divide this file into chunks. Copypasta code I don't understand.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
n = split_n

#Split list of runs and save them in the proper directory
x = list(divide_chunks(file,n))
for i in range(0,len(x)):
    desired_string = "'"
    for run in x[i][:-1]:
        desired_string = desired_string + str(int(run)) + ', '
    desired_string = desired_string + str(int(x[i][-1])) + "'"
    np.savetxt(homeDirectory+'/TriggerResults/'+AcqDet+'/WSRunsFolder/GoodFormat/'+AcqDet+'_goodformat_part'+str(i+1)+'.txt',[desired_string],fmt="%s")

#for every partial run list write a dc query for it and save it in the DCQueries folder
path_to_dc_query_folder = homeDirectory+'/TriggerResults/'+AcqDet+"/DCQueries/" 
n_partial_runlists = len(os.listdir(homeDirectory+'/TriggerResults/'+AcqDet+'/WSRunsFolder/GoodFormat/')) #Get the number of partial run lists
for i in range(0,n_partial_runlists  ): #for each one
	run_list_file = open((homeDirectory+'/TriggerResults/'+AcqDet+'/WSRunsFolder/GoodFormat/'+AcqDet+'_goodformat_part'+str(i+1)+'.txt'))
	run_list = run_list_file.read() #load in the partial run list
	print(run_list)
	#write the yaml query
	with open(path_to_dc_query_folder + AcqDet +'.DCquery_part'+str(i+1)+'.yaml','w') as f:
	    f.write("usecase: 'DataRQ'\n")
	    f.write("software_versions: " +  "'"+lzapversion+"'"+'\n')
	    f.write("run: "+ run_list)
	    f.close()
        
#We go through the run list and grab DC info


import DataCatalog
from datetime import date
today = date.today()
runlist = [int(x) for x in np.loadtxt(poorformatrunlist_path,ndmin=1)]
total_runs = len(runlist)
software_version = lzapversion
#separate the query for these tags into their own data catalog
begin_times_list = []
end_times_list = []
n_rq_files_directory_list = []
n_rq_files_catalog_list  = []
rq_directory_list = []
n_rq_events_directory_list = []
n_rq_events_catalog_list = []

n_raw_files_directory_list  = []
n_raw_files_catalog_list  = []
raw_directory_list   = []
n_raw_events_directory_list = []
n_raw_events_catalog_list = []
noe_catalog = DataCatalog.Catalog(tags=['number_events'])
software_versions_catalog = DataCatalog.Catalog(tags=['software_versions'])
url_catalog = DataCatalog.Catalog(tags=['url'])
begin_end_time_catalog = DataCatalog.Catalog(tags=['begin_time','end_time'])
catalog = DataCatalog.Catalog(tags=['number_events','software_versions','url','begin_time']) #Select the number_events tag from the catalog
for run in runlist: #For every run
    #Set up our catalog instances
    try:
        rq_noe_selection = noe_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)
        rq_software_versions_selection = software_versions_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)
        rq_url_selection = url_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)
        rq_begin_end_time_selection = begin_end_time_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)

        #Gets begin times and end times for each run
        begin_times = list(set([x[0] for x in rq_begin_end_time_selection]))[0]
        end_times = list(set([x[1] for x in rq_begin_end_time_selection]))[-1]
        begin_times_list = begin_times_list + [begin_times]
        end_times_list = end_times_list + [end_times]


        subselection_rq_noe_selection = noe_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)
        subselection_rq_software_versions_selection = software_versions_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)
        subselection_rq_url_selection = url_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)
        subselection_rq_begin_end_time_selection = begin_end_time_catalog.select('run',run).select('usecase','DataRQ').select('software_versions',software_version)


        directory = [x[:-24] for x in subselection_rq_url_selection][0]
        rq_directory_list = rq_directory_list + [directory]


        filelist = [x for x in os.listdir(directory) if x.endswith('.root')]
        n_rq_files_directory_list  = n_rq_files_directory_list + [len(filelist)]

        n_rq_files = len(list(subselection_rq_url_selection))# grabs number of rq files from RQ selection
        n_rq_files_catalog_list = n_rq_files_catalog_list + [n_rq_files]


        #Gets number of RQ events in the first file    
        n_rq_events_first_file = [float(x) for x in subselection_rq_noe_selection ][0]
        n_rq_events_directory_list = n_rq_events_directory_list + [n_rq_files*n_rq_events_first_file]

        n_rq_events_catalog_list = n_rq_events_catalog_list + [sum([float(x) for x in subselection_rq_noe_selection ])] #Sum the number of events in each file of the run


        subselection_raw_noe_selection = noe_catalog.select('run',run).select('usecase','DataRaw')
        subselection_raw_url_selection = url_catalog.select('run',run).select('usecase','DataRaw')
        subselection_raw_begin_end_time_selection = begin_end_time_catalog.select('run',run).select('usecase','DataRaw')



        try:
            directory = [ x[:-38] for x in subselection_raw_url_selection][0]
            raw_directory_list = raw_directory_list + [directory]

            filelist = [x for x in os.listdir(directory) if x.endswith('.root')]
            n_raw_files_directory = len(filelist)
            n_raw_files_directory_list = n_raw_files_directory_list + [n_raw_files_directory]


            n_raw_files = len(subselection_raw_noe_selection)
            n_raw_files_catalog_list = n_raw_files_catalog_list + [n_raw_files]

            n_raw_events_first_file = [float(x) for x in subselection_raw_noe_selection][0]

            n_raw_events_directory_list = n_raw_events_directory_list + [n_raw_events_first_file*n_raw_files_directory]

            n_raw_events_catalog_list = n_raw_events_catalog_list + [np.sum([float(x) for x in subselection_raw_noe_selection])]


        except:
            None    
            print('Weirdness in run ' + str(run))
    except:
        begin_times_list += ['error']
        end_times_list += ['error']
        n_rq_files_directory_list += [-1]
        n_rq_files_catalog_list  += [-1]
        rq_directory_list += ['error']
        n_rq_events_directory_list += [-1]
        n_rq_events_catalog_list += [0] #Switched from -1 to avoid giving false frac_processed info

        n_raw_files_directory_list  += [-1]
        n_raw_files_catalog_list  += [-1]
        raw_directory_list   += ['error']
        n_raw_events_directory_list += [-1]
        n_raw_events_catalog_list += [-1]
datadict = {}
datadict['runs'] = runlist
datadict['Date_Accessed'] = today
datadict['software_version'] = software_version

datadict['begin_times'] = begin_times_list
datadict['end_times'] = end_times_list
datadict['n_rq_files_directory'] = n_rq_files_directory_list
datadict['n_rq_files_catalog'] = n_rq_files_catalog_list 
datadict['rq_directory'] = rq_directory_list
datadict['n_rq_events_directory'] = n_rq_events_directory_list
datadict['n_rq_events_catalog'] = n_rq_events_catalog_list

datadict['n_raw_files_directory'] = n_raw_files_directory_list 
datadict['n_raw_files_catalog']  = n_raw_files_catalog_list 
datadict['raw_directory'] = raw_directory_list  
datadict['n_raw_events_directory'] = n_raw_events_directory_list
datadict['n_raw_events_catalog'] = n_raw_events_catalog_list
datadict['frac_processed'] = np.array(datadict['n_rq_events_catalog'])/np.array(datadict['n_raw_events_directory'])

np.save(homeDirectory+'/TriggerResults/'+AcqDet+'/Results/DC_Dict.npy',datadict)
       #We then go back to the shell script to query the Data Catalog
