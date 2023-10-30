#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys,math
from scipy.stats import binned_statistic
pd.options.mode.chained_assignment = None

def trackPercent(place,totalLength,strLen): #percent output tracker
    percent = place/totalLength*100
    string="{:.2f} % complete".format(percent)
    sys.stdout.write("\r") #this "moves the cursor" to the beginning of the I0 line
    sys.stdout.write(" "*strLen) #this "clears" whatever was on the line last time by writing whitespace
    sys.stdout.write("\r") #move the cursor back to the start again
    sys.stdout.write(string) #display the current percent we are at
    sys.stdout.flush() #flush finishes call to print() (this is like what's under the hood of print function)
    strLen=len(string) #return the new string length for next function call
    return strLen

# data prep / cleaning
def genNBDataMCMC(cleanFile="../MCMC/data/combined_cleaned.csv",bins=5):
    df = pd.read_csv(cleanFile)
    strLen = 0
    columns2ignore = ["rotation","singlePeak","doublePeak"]
    #algorithm -- 
    #1. generate bins for each parameter
    #2. go through dataframe, find bin number for each parameter at every entry
    #3. each entry will be of the shape (binNumber1,binNumber2....binNumerN),label
    print("Binning data...\n")
    for (i,c) in enumerate(df.columns):
        if c in columns2ignore:
            continue
        else:
            x = np.linspace(np.min(df[c]),np.max(df[c]),len(df[c]))
            sum,binEdges,binNumbers = binned_statistic(df[c],df[c],statistic='sum',bins=bins) 
            df[c] = binNumbers #replace data with bin numbers
        strLen = trackPercent(i,len(df.columns),strLen)
    print("\nGenerating labels...\n")
    strLen = 0
    df["label"] = ['']*len(df)
    for i in range(len(df)):
        label = "singlePeak" if df["singlePeak"][i] == 1 else "doublePeak"
        df["label"][i] = label
        strLen = trackPercent(i,len(df),strLen)
    df.drop(columns=columns2ignore,inplace=True)
    print("\nSaving data...\n")
    df.to_csv("NBMCMCData.csv",index=False)

def genNBDataSDSS(cleanFile="../SDSS/combined_noSize_cleaned.csv",bins=5):
    df = pd.read_csv(cleanFile)
    strLen = 0
    if "noSize" in cleanFile:
        columns2ignore = ["specObjID","plate","ra","dec","mjd","fiberid","run2d","objid",
                        "fieldID","redshiftError","spectroSynFluxIvar_u","spectroSynFluxIvar_g",
                        "spectroSynFluxIvar_r","spectroSynFluxIvar_z","spectroSynFluxIvar_i",
                        "err_u","err_g","err_r","err_i","err_z"]
    else:
        columns2ignore = ["specObjID","plate","ra","dec","mjd","fiberid","run2d","objid",
                        "fieldID","redshiftError","spectroSynFluxIvar_u","spectroSynFluxIvar_g",
                        "spectroSynFluxIvar_r","spectroSynFluxIvar_z","spectroSynFluxIvar_i",
                        "err_u","err_g","err_r","err_i","err_z","petroRadErr_u","petroRadErr_g",
                        "petroRadErr_r","petroRadErr_z","petroRadErr_i"]
    print("Reformatting input for NB")
    print("Binning data...\n")
    for (i,c) in enumerate(df.columns):
        if c in columns2ignore or c == "class":
            continue
        else:
            x = np.linspace(np.min(df[c]),np.max(df[c]),len(df[c]))
            sum,binEdges,binNumbers = binned_statistic(df[c],df[c],statistic='sum',bins=bins) 
            df[c] = binNumbers #replace data with bin numbers
        strLen = trackPercent(i,len(df.columns),strLen)
    df.drop(columns=columns2ignore,inplace=True)
    print("\nsaving data")
    out = "NBSDSS_noSizeData.csv" if "noSize" in cleanFile else "NBSDSS_wSizeData.csv"
    df.to_csv(out,index=False)

def genSDSSweights(cleanFile="../SDSS/combined_noSize_cleaned.csv"):
    df = pd.read_csv(cleanFile)
    if "noSize" in cleanFile:
        weightColumns = ["spectroSynFluxIvar_u","spectroSynFluxIvar_g",
                        "spectroSynFluxIvar_r","spectroSynFluxIvar_z","spectroSynFluxIvar_i",
                        "err_u","err_g","err_r","err_i","err_z"]
    else:
        weightColumns = ["spectroSynFluxIvar_u","spectroSynFluxIvar_g",
                        "spectroSynFluxIvar_r","spectroSynFluxIvar_z","spectroSynFluxIvar_i",
                        "err_u","err_g","err_r","err_i","err_z","petroRadErr_u","petroRadErr_g",
                        "petroRadErr_r","petroRadErr_z","petroRadErr_i"]
    weights = df[weightColumns]
    for (i,c) in enumerate(weights.columns):
        if "Ivar" in c:
            weights[c] = np.sqrt(weights[c])
            weights[c] = weights[c]/np.max(weights[c]) 
        else:
            weights[c] = 1/weights[c]
            weights[c] = weights[c]/np.max(weights[c])
    weights = weights.sum(axis=1) 
    weights = weights/np.max(weights)
    return weights
    