#!/bin/bash 
# Author: Marcus Converse 
# This script lives in my the folder triggerefficiencyexternal trigger in my home directory
# This script needs to:

# Generate a directory structure in my NERSC home directory

# Take a poorly formatted list of runs copied from the acquisitions spreadsheet
# Properly format it and split it into lists of split_n runs each
# Query the data catalog with these run lists and save the catalog queries
# Find all the files from previously running ALPACA
# Combine these outputs into a single data file and generate our efficiency report

AcqDet='Test'
scratchDirectory='/pscratch/sd/m/mconver2'
homeDirectory='/global/homes/m/mconver2/'
alpacaDirectory='/global/homes/m/mconver2/ALPACA'
lzBuildPath='/global/homes/m/mconver2/.local/share/jupyter/kernels/lz-nersc-jupyter/lzBuild.sh'
triggerefficiencyexternaltriggerPath='/global/homes/m/mconver2/triggerefficiencyexternaltrigger/'
poorformatrunlist_path='/global/homes/m/mconver2/triggerefficiencyexternaltrigger/VerificationFolder/TestRuns.txt' #path to the poorly formatted run list
WIMPSEARCHDATA='False'
split_n=5
binwidth=10
lowerpulsearealimit=0
upperpulsearealimit=2400
SetNominalThreshold=200 
#Radius/Inefficiency stuff
#radius cutoffs 0-40,40-57,57-68 are *roughly* equal in area
innerradius0=0 #Center of detector
outerradius0=40 
innerradius1=40
outerradius1=57
innerradius2=57
outerradius2=80 
lzapversion='LZAP-5.8.0'

#Generating the directory structure

#Revision makes these directories in user home directory

mkdir $homeDirectory/TriggerResults/
mkdir $homeDirectory/TriggerResults/$AcqDet || :
mkdir $homeDirectory/TriggerResults/$AcqDet/WSRunsFolder || :
cp $poorformatrunlist_path $homeDirectory/TriggerResults/$AcqDet/WSRunsFolder/$AcqDet.txt
mkdir $homeDirectory/TriggerResults/$AcqDet/WSRunsFolder/GoodFormat || :
mkdir $homeDirectory/TriggerResults/$AcqDet/DCQueries || :
mkdir $homeDirectory/TriggerResults/$AcqDet/InputLists || :
mkdir $homeDirectory/TriggerResults/$AcqDet/Results || :
#And the one directory on scratch
mkdir $scratchDirectory/TriggerEfficiencyExtTrigger/$AcqDet/ || :

pfrlp=$homeDirectory/TriggerResults/$AcqDet/WSRunsFolder/$AcqDet.txt

# Running a python script to:
# Properly format and split runs
# Write DC queries with the split run lists to the DCQueries folder

export pfrlp
export split_n
export AcqDet
export scratchDirectory
export WIMPSEARCHDATA
export binwidth
export lowerpulsearealimit
export upperpulsearealimit
export SetNominalThreshold 
#Radius/Inefficiency stuff
#radius cutoffs 0-40,40-57,57-68 are *roughly* equal in area
export innerradius0 #Center of detector
export outerradius0 
export innerradius1
export outerradius1
export innerradius2
export outerradius2 
export lzapversion
export homeDirectory
# source $lzBuildPath
# python $triggerefficiencyexternaltriggerPath/RunListParser.py



#Run a python script collating all of this output
source $lzBuildPath

#python PythonScript1.py
python $triggerefficiencyexternaltriggerPath/TriggerEfficiencyProcessor.py