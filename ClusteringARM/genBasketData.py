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

def genBasketDataMCMC(cleanFile="../MCMC/data/combined_cleaned.csv",outFile="MCMCbasket.csv"):
    df = pd.read_csv(cleanFile)
    with open(outFile,"w") as f:
        f.write("peak,i,fDom\n")
    strLen = 0
    for i in range(len(df)):
        fvals = [df.f1[i],df.f2[i],df.f3[i],df.f4[i]]
        flabels = ["f1","f2","f3","f4"]
        maxInd = np.argmax(fvals)
        fDom = flabels[maxInd]
        if df.singlePeak[i] == True:
            peak = "singlePeak"
        elif df.doublePeak[i] == True:
            peak = "doublePeak"
        else:
            peak = "singlePeak,doublePeak"
        if df["i"][i] < 30:
            inc = "low_i"
        elif df["i"][i]>30 and df["i"][i]<60:
            inc = "med_i"
        else:
            inc = "high_i"
        with open(outFile,"a") as f:
            f.write(f"{peak},{inc},{fDom}\n")
        strLen = trackPercent(i,len(df),strLen)
    return 0

def genBasketDataSDSS(cleanFile="../SDSS/combined_noSize_cleaned.csv",outFile="SDSS_noSizeBasket.csv",size=False):
    df = pd.read_csv(cleanFile)
    with open(outFile,"w") as f:
        if size:
            f.write("magDom,spectroFluxDom,class,distance,sizeDom\n")
        else:
            f.write("magDom,spectroFluxDom,class,distance\n")
    strLen = 0
    for i in range(len(df)):
        magVals = [df.u[i],df.g[i],df.r[i],df.i[i],df.z[i]]
        magLabels = ["u","g","r","i","z"]
        maxInd = np.argmax(magVals)
        magDom = magLabels[maxInd]
        spectroFluxVals = [df.spectroSynFlux_u[i],df.spectroSynFlux_g[i],df.spectroSynFlux_i[i],df.spectroSynFlux_z[i]]
        spectroFluxLabels = ["spectroSynFlux_u","spectroSynFlux_g","spectroSynFlux_i","spectroSynFlux_z"]
        maxInd = np.argmax(spectroFluxVals)
        spectroFluxDom = spectroFluxLabels[maxInd]
        c = df["class"][i]
        if df.redshift[i] < 1:
            d = "close"
        else:
            d = "far"
        if size:
            sizeVals = [df.petroRad_u[i],df.petroRad_g[i],df.petroRad_r[i],df.petroRad_i[i],df.petroRad_z[i]]
            sizeLabels = ["petroRad_u","petroRad_g","petroRad_r","petroRad_i","petroRad_z"]
            maxInd = np.argmax(sizeVals)
            sizeDom = sizeLabels[maxInd]
            with open(outFile,"a") as f:
                f.write(f"{magDom},{spectroFluxDom},{c},{d},{sizeDom}\n")
        else:
            with open(outFile,"a") as f:
                f.write(f"{magDom},{spectroFluxDom},{c},{d}\n")
        strLen = trackPercent(i,len(df),strLen)
    return 0

def main(MCMCfile="../MCMC/data/combined_cleaned.csv",SDSS_noSizeFile="../SDSS/combined_noSize_cleaned.csv",SDSS_sizeFile="../SDSS/combined_wSize_cleaned.csv"):
    print("generating basket data for MCMC")
    genBasketDataMCMC(MCMCfile)
    print("\ngenerating basket data for SDSS_noSize")
    genBasketDataSDSS(SDSS_noSizeFile)
    print("\ngenerating basket data for SDSS_wSize")
    genBasketDataSDSS(SDSS_sizeFile,outFile="SDSS_wSizeBasket.csv",size=True)
    return 0
        
main() #note, takes ~10 minutes to run on my machine
        



        

