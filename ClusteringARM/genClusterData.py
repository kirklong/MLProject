#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys,math

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


def genClusterDataMCMC(cleanFile="../MCMC/data/combined_cleaned.csv"):
    df = pd.read_csv(cleanFile)
    with open("MCMC_singlePeak_cluster.csv","w") as f: #overwrite and empty if already exists
        pass
    with open("MCMC_doublePeak_cluster.csv","w") as f:
        pass
    strLen = 0
    for i in range(len(df)):
        fvals = [df.f1[i],df.f2[i],df.f3[i],df.f4[i]]
        fvals /= np.sum(fvals) #normalize
        if df.singlePeak[i] == True:
            with open("MCMC_singlePeak_cluster.csv","a") as f:
                f.write(f"{df['i'][i]},{df['r̄'][i]},{df.Mfac[i]},{df.rFac[i]},{fvals[0]},{fvals[1]},{fvals[2]},{fvals[3]},{df.pa[i]},{df['Sα'][i]},{df.phaseAmplitude[i]},{df.t_char[i]},{df.rBLR[i]},{df.FHWM[i]}\n")
        else:
            with open("MCMC_doublePeak_cluster.csv","a") as f:
                f.write(f"{df['i'][i]},{df['r̄'][i]},{df.Mfac[i]},{df.rFac[i]},{fvals[0]},{fvals[1]},{fvals[2]},{fvals[3]},{df.pa[i]},{df['Sα'][i]},{df.phaseAmplitude[i]},{df.t_char[i]},{df.rBLR[i]},{df.FHWM[i]}\n")
        strLen = trackPercent(i,len(df),strLen)
    return 0

def genClusterDataSDSS(cleanFile="../SDSS/combined_noSize_cleaned.csv",size=False):
    labels = ["STAR","GALAXY","QSO"]
    if size:
        outFiles = ["SDSS_wSize_{}_cluster.csv".format(label) for label in labels]
    else:
        outFiles = ["SDSS_noSize_{}_cluster.csv".format(label) for label in labels]
    for outFile in outFiles:
        with open(outFile,"w") as f: #overwrite and empty if already exists
            pass            
    df = pd.read_csv(cleanFile)
    strLen = 0
    for i in range(len(df)):
        row = df.iloc[i]
        c = row["class"]
        if size:
            outFile = "SDSS_wSize_{}_cluster.csv".format(c)
            out = np.hstack((row[2],row[9:13],row[18:25],row[-10:-5])) #redshift, magnitudes, ra/dec, sizes
            with open(outFile,"a") as f:
                f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}\n".format(*out))
        else:
            outFile = "SDSS_noSize_{}_cluster.csv".format(c)
            out = np.hstack((row[2],row[9:13],row[18:25])) #redshift, magnitudes, ra/dec
            with open(outFile,"a") as f:
                f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n".format(*out))
        strLen = trackPercent(i,len(df),strLen)
    return 0

def main(MCMCfile="../MCMC/data/combined_cleaned.csv",SDSS_noSizeFile="../SDSS/combined_noSize_cleaned.csv",SDSS_sizeFile="../SDSS/combined_wSize_cleaned.csv"):
    print("generating cluster data for MCMC")
    genClusterDataMCMC(MCMCfile)
    print("\ngenerating cluster data for SDSS_noSize")
    genClusterDataSDSS(SDSS_noSizeFile)
    print("\ngenerating cluster data for SDSS_wSize")
    genClusterDataSDSS(SDSS_sizeFile,size=True)
    return 0

main()

def genSamples(fileList,n=1000):
    for file in fileList:
        out = file[:-4]+"_samples.csv"
        with open(file,"r") as f:
            lines = f.readlines()
        randlines = np.random.randint(0,len(lines),size=n)
        with open(out,"w") as f:
            for line in randlines:
                f.write(lines[line])
    return 0
