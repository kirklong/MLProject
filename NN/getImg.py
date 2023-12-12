#!/usr/bin/env python
import os, sys, math
import pandas as pd
import numpy as np

def trackPercent(place,totalLength,strLen): #percent output tracker
    percent = place/totalLength*100
    if math.floor(percent)==69:
        string="{:.2f} % complete -- nice".format(percent)
    else:
        string="{:.2f} % complete".format(percent)
    sys.stdout.write("\r") #this "moves the cursor" to the beginning of the I0 line
    sys.stdout.write(" "*strLen) #this "clears" whatever was on the line last time by writing whitespace
    sys.stdout.write("\r") #move the cursor back to the start again
    sys.stdout.write(string) #display the current percent we are at
    sys.stdout.flush() #flush finishes call to print() (this is like what's under the hood of print function)
    strLen=len(string) #return the new string length for next function call
    return strLen

path = os.path.join(os.getcwd(),'../SDSS')
sys.path.append(path)

from query import getImg, getSpectrum

#collect sample of quasar, galaxy, and star images to 
#train neural network on identifying 

#can also do this with spectra

def downloadIMGs(n,objType, dataFile="../SDSS/combined_wSize_cleaned.csv",opt="",width=512,height=512,scale=0.4):
    #note -- most of the images are BAD
    data = pd.read_csv(dataFile)
    IDs = data[data['class'] == objType]['specObjID'].sample(n).values
    RAs = data[data['class'] == objType]['ra'].sample(n).values
    DECs = data[data['class'] == objType]['dec'].sample(n).values
    strLen = 0; errCount = 0; errList = []
    for i, (id,ra,dec) in enumerate(zip(IDs,RAs,DECs)):
        d = dict(specObjID=id,ra=ra,dec=dec)
        status,out = getImg(d,opt=opt,width=width,height=height,scale=scale)
        if status != 0:
            print("Error with object: ",id)
            print(out)
            errCount += 1
            errList.append(id)
            os.remove("images/image_{}.jpeg".format(id))
        strLen = trackPercent(i+1,n,strLen)
    print("\n")
    print("Errors: {}/{} = {:.2g}%".format(errCount,n,errCount/n*100))
    return errList

def downloadSpectra(n,objType, dataFile="../SDSS/combined_wSize_cleaned.csv",format="csv"):
    data = pd.read_csv('../SDSS/combined_wSize_cleaned.csv')
    IDs = data[data['class'] == objType]['specObjID'].sample(n).values
    plates = data[data['class'] == objType]['plate'].sample(n).values
    dates = data[data['class'] == objType]['mjd'].sample(n).values
    fibers = data[data['class'] == objType]['fiberid'].sample(n).values
    runs = data[data['class'] == objType]['run2d'].sample(n).values
    strLen = 0; errList = []; errCount = 0 
    for i,(id,plate,date,fiber,run) in enumerate(zip(IDs,plates,dates,fibers,runs)):
        d = dict(specObjID=id,plate=plate,mjd=date,fiberid=fiber,run2d=run)
        status,out = getSpectrum(d,format=format)
        if status != 0:
            print("Error with object: ",id)
            print(out)
            errCount += 1
            errList.append(id)
            os.remove("spectra/spectrum_{}.{}".format(id,format))
        strLen = trackPercent(i+1,n,strLen)
    print("\n")
    print("Errors: {}/{} = {:.2g}%".format(errCount,n,errCount/n*100))
    return errList
    

