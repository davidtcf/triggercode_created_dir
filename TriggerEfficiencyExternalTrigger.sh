#!/bin/bash 
# Author: Marcus Converse 
# This script lives in my the folder triggerefficiencyexternal trigger in my home directory
# This script needs to:

# Generate a directory structure in my NERSC home directory

# Take a poorly formatted list of runs copied from the acquisitions spreadsheet
# Properly format it and split it into lists of split_n runs each
# Query the data catalog with these run lists and save the catalog queries
# Run ALPACA over each of these run lists
# Combine these outputs into a single data file and generate our efficiency report

# AcqDet='preSR31300x6_test'
# #alpacaDirectory='/global/homes/m/mconver2/ALPACA'
# #lzBuildPath='/global/homes/m/mconver2/.local/share/jupyter/kernels/lz-nersc-jupyter/lzBuild.sh'
# #triggerefficiencyexternaltriggerPath='/global/homes/m/mconver2/triggerefficiencyexternaltrigger/'
# poorformatrunlist_path='/global/homes/m/mconver2/PreSR3_0428_RandomTriggerTest_1300x6_20231005.txt' #path to the poorly formatted run list
# WIMPSEARCHDATA='False'
# split_n=5
# binwidth=10
# lowerpulsearealimit=0
# upperpulsearealimit=2400
# SetNominalThreshold=200 
# #Radius/Inefficiency stuff
# #radius cutoffs 0-40,40-57,57-68 are *roughly* equal in area
# innerradius0=0 #Center of detector
# outerradius0=40 
# innerradius1=40
# outerradius1=57
# innerradius2=57
# outerradius2=80 
# lzapversion='LZAP-5.8.0'

#Generating the directory structure

#Revision makes these directories in user home directory

mkdir $WORKDIR/TriggerResults/
mkdir $WORKDIR/TriggerResults/$AcqDet || :
mkdir $WORKDIR/TriggerResults/$AcqDet/WSRunsFolder || :
cp $poorformatrunlist_path $WORKDIR/TriggerResults/$AcqDet/WSRunsFolder/$AcqDet.txt
mkdir $WORKDIR/TriggerResults/$AcqDet/WSRunsFolder/GoodFormat || :
mkdir $WORKDIR/TriggerResults/$AcqDet/DCQueries || :
mkdir $WORKDIR/TriggerResults/$AcqDet/InputLists || :
mkdir $WORKDIR/TriggerResults/$AcqDet/Results || :
#And the one directory on scratch
mkdir $SCRATCH/TriggerEfficiencyExtTrigger/$AcqDet/ || :

pfrlp=$WORKDIR/TriggerResults/$AcqDet/WSRunsFolder/$AcqDet.txt

# Running a python script to:
# Properly format and split runs
# Write DC queries with the split run lists to the DCQueries folder

export pfrlp
# export split_n
# export AcqDet
# export scratchDirectory
# export WIMPSEARCHDATA
# export binwidth
# export lowerpulsearealimit
# export upperpulsearealimit
# export SetNominalThreshold 
# #Radius/Inefficiency stuff
# #radius cutoffs 0-40,40-57,57-68 are *roughly* equal in area
# export innerradius0 #Center of detector
# export outerradius0 
# export innerradius1
# export outerradius1
# export innerradius2
# export outerradius2 
# export lzapversion
# export homeDirectory
source $lzBuildPath
python $triggerefficiencyexternaltriggerPath/RunListParser.py




#Query the data catalog with these run lists and save the catalog queries.

#Do LZ Build
source $alpacaDirectory/setup.sh
build
#Copy TriggerEfficiencyExtTrigger to ALPACA modules directory and add with module helper
cp -a $triggerefficiencyexternaltriggerPath/TriggerEfficiencyExtTrigger $alpacaDirectory/modules/TriggerEfficiencyExtTrigger
moduleHelper TriggerEfficiencyExtTrigger --add


count=$(ls -l $WORKDIR/TriggerResults/$AcqDet/DCQueries | grep ^- | wc -l)
START=1
END=$count
for (( i=$START; i<=$END; i++ ))
do
	echo $WORKDIR/TriggerResults/$AcqDet/DCQueries/$AcqDet.DCquery_part$i.yaml
	queryData $WORKDIR/TriggerResults/$AcqDet/DCQueries/$AcqDet.DCquery_part$i.yaml -o $WORKDIR/TriggerResults/$AcqDet/InputLists/$AcqDet.InputList_part$i.list
done





#Running ALPACA over each file list


source $alpacaDirectory/setup.sh #build ALPACA again because we just added a module
build

for (( i=$START; i<=$END; i++ ))
do
	cp $WORKDIR/TriggerResults/$AcqDet/InputLists/$AcqDet.InputList_part$i.list $alpacaDirectory/modules/TriggerEfficiencyExtTrigger/inputs/TriggerEfficiencyExtTriggerInputFiles.list #Copy the input list to the proper place
	TriggerEfficiencyExtTrigger -w #Run the module

	cp $alpacaDirectory/run/TriggerEfficiencyExtTrigger/TriggerEfficiencyExtTriggerAnalysis.root $SCRATCH/TriggerEfficiencyExtTrigger/$AcqDet/$AcqDet.part$i.root #copy ALPACA output to scratch

	rm $alpacaDirectory/run/TriggerEfficiencyExtTrigger/TriggerEfficiencyExtTriggerAnalysis.root #clear the file
done

moduleHelper TriggerEfficiencyExtTrigger --remove
rm -rf $alpacaDirectory/modules/TriggerEfficiencyExtTrigger
#Run a python script collating all of this output
source $lzBuildPath

#python PythonScript1.py
python $triggerefficiencyexternaltriggerPath/TriggerEfficiencyProcessor.py