# TriggerEfficiencyExternalTrigger

## Getting started running this pipeline

This pipeline is used to measure the trigger efficiency from random/GPS triggers for a specified dataset. 

The repository should be cloned into the home directory of the user, as many paths are by default specified in reference to the home directory of the user.
The top level scripts the user interacts with are the configuration script for environment variables, "TriggerEfficiencyExternalTrigger.sh" and "NoALPACATriggerEfficiencyExternalTrigger.sh". 

The user should specify environment variables and paths to data sources/output directories in a script, triggerconf.sh. 

This code should be run within the shifter image indicated in the "Some magic" section of the following link using the command shifterCOS7 bash
https://luxzeplin.gitlab.io/docs/softwaredocs/computing/usdc/shifter.html

Once dependencies are installed, the script as cloned should be good to run in a shifter image over the data set indicated by TestRuns.txt and produce the output in the folder Test.

### Dependencies

This code is dependent on having the LZ build installed. Instructions on installing the LZ kernel can be found here: https://luxzeplin.gitlab.io/docs/softwaredocs/computing/usdc/python.html


The python scripts are dependent on the libraries which should be added to your LZ python kernel if they aren't already there:

numpy

DataCatalog

datetime

uproot

matplotlib

mpmath

pandas

scipy

pylab

tqdm

pickle

numba

time





### Environment variables

alpacaDirectory specifies the path to where ALPACA is installed for the user.

lzBuildPath specifies the path to the script lzBuild.sh, for me this is in the jupyter kernel folder.

triggerefficiencyexternaltriggerPath specifies the path to the folder the git repository is cloned to.

AcqDet specifies the acquisition details for your dataset. This is, essentially, a label for the data set that will be used to determine the names of folders where output figures will be generated.

poorformatrunlist_path specifies the path to the text file where the poorly formatted list of runs you wish to include in your dataset is saved. More on this later.

WIMPSEARCHDATA is a string that should be set to "True" or "False". If the data set is from the WIMP search, or S2 triggered in some way, it should be set to "True". If the data set is not S2-triggered, it should be set to "False".

split_n is an integer that specifies the number of runs in one processing chunk. 5 is a good value for this.

binwidth currently does nothing

lowerpulsearealimit currently does nothing

upperpulsearealimit currently does nothing

SetNominalThreshold sets the nominal threshold that the script will report not-triggered pulses relative to. For SR3, this is set to 200.

inner/outerradius0-2 set the bounds of concentric annuli that the script reports the number of total pulses above nominal threshold for, as well as the largest not-triggered pulse within.

lzapversion is a string allowing the user to specify the LZAP version used in data catalog queries.

## User Input Files

The poorformatrunlist is the only input file the user needs to generate manually. This is a text file containing a list of runs that the user would like to process through this pipeline. The format should be one run per line of the text file. This one run per line format is used as one can directly copy run IDs from the acquisitions spreadsheet and paste them into the text file.

## How does this pipeline function?

This pipeline is a shell script that take user specified environment variables and a list of runs specified as above and interfaces with the python script RunListParser.py, the Data Catalog, then ALPACA, then TriggerEfficiencyProcessor.py. 

The user should first specify environment variables in a shell script like triggerconf_example.sh and run that to set up for running TriggerEfficiencyExternalTrigger.sh.

TriggerEfficiencyExternalTrigger.sh first generates a directory structure to save the output of the python scripts and ALPACA. 

The list of runs to be processed is then passed to RunListParser.py. RunListParser.py generates Data Catalog queries for the list of runs, then queries the data catalog for some diagnostic information (run start/stop times, number of events, paths to raw data, etc.). The Data Catalog queries are saved in the DCQueries folder in the output directory. The diagnostic information is saved in the Results folder as DC_Dict.npy as a dictionary in python. 

The shell script then copies the folder TriggerEfficiencyExtTrigger into the user's ALPACA/modules directory, and adds the module to ALPACA.

The shell script then queries the Data Catalog using the queries in the DCQueries folder, saving the results of the queries in the InputLists folder.

The shell script then, in series, copies each input file list from InputLists to the proper input folder in ALPACA and runs the module TriggerEfficiencyExtTrigger over the file list, saving the output in the user's scratch directory.

When all files are processed, the script removes TriggerEfficiencyExtTrigger from the ALPACA module list, and runs TriggerEfficiencyProcessor.py over the ALPACA output.

## What about NoALPACATriggerEfficiencyExternalTrigger.sh?

This is for when you've already processed your data through ALPACA and want to reprocess it through TriggerEfficiencyProcessor.py without all the previous steps. It takes the same inputs as before, but skips RunlistParser.py, the Data Catalog Queries, and running ALPACA, directly feeding the already processed ALPACA output from the user home directory to TriggerEfficiencyProcessor.py.

## What commands should I run for this pipeline?
Install your dependencies, specify your environment variables in the configuration script, and run these two commands:

. triggerconf_example.sh

. TriggerEfficiencyExternalTrigger.sh
