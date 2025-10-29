#!/bin/bash 
# Author: Marcus Converse 
#This script lives in my home directory and is edited to set up environment variables for the TriggerEfficiencyExternalTrigger and NoALPACATriggerEfficiencyExternalTrigger.sh scripts.

#These are environment variables that you set when you install the trigger code for the first time
alpacaDirectory="$WORKDIR/alpaca"
lzBuildPath="$WORKDIR/lz-nersc-jupyter/lz-standard/lzBuild.sh"
triggerefficiencyexternaltriggerPath="$WORKDIR/triggerefficiencyexternaltrigger"
export alpacaDirectory
export lzBuildPath
export triggerefficiencyexternaltriggerPath


#These are environment variables you set before you run the code.
AcqDet='SumTrigger-Test2'
poorformatrunlist_path="$triggerefficiencyexternaltriggerPath/VerificationFolder/AugustSumTrigger.txt" #path to the poorly formatted run listk
WIMPSEARCHDATA='False'
split_n=5
binwidth=10
lowerpulsearealimit=0
upperpulsearealimit=2400
SetNominalThreshold=200 
innerradius0=0 #Center of detector
outerradius0=40 
innerradius1=40
outerradius1=57
innerradius2=57
outerradius2=80 
lzapversion='LZAP-6.2.1'

export split_n
export AcqDet
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



echo "Acquisition Details: $AcqDet"
echo "Path to list of runs: $poorformatrunlist_path"
echo "Is this WIMP Search Data: $WIMPSEARCHDATA"
echo "Which LZAP Version: $lzapversion"
