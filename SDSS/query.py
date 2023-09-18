#!/usr/bin/env python
import sys
import os
import re
import pandas as pd
import numpy as np
import math

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

SQL_query_QSO = """SELECT TOP {0}
s.specObjID, s.class, s.z as redshift, s.zErr as redshiftError, s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_z, s.spectroSynFluxIvar_i, s.spectroSynFlux_i, s.spectroSynFlux_z, s.spectroSynFlux_u, s.spectroSynFlux_g, s.plate, s.mjd, s.fiberid, s.run2d, p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.fieldID, p.err_u, p.err_g, p.err_r, p.err_i, p.err_z, p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, p.petroRadErr_u, p.petroRadErr_g, p.petroRadErr_r, p.petroRadErr_z, p.petroRadErr_i
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   (s.class = 'QSO') AND (zWarning = 0)"""

SQL_query_GAL = """SELECT TOP {0}
s.specObjID, s.class, s.z as redshift, s.zErr as redshiftError, s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_z, s.spectroSynFluxIvar_i, s.spectroSynFlux_i, s.spectroSynFlux_z, s.spectroSynFlux_u, s.spectroSynFlux_g, s.plate, s.mjd, s.fiberid, s.run2d, p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.fieldID, p.err_u, p.err_g, p.err_r, p.err_i, p.err_z, p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, p.petroRadErr_u, p.petroRadErr_g, p.petroRadErr_r, p.petroRadErr_z, p.petroRadErr_i
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   (s.class = 'GALAXY') AND (zWarning = 0)"""

SQL_query_STAR = """SELECT TOP {0}
s.specObjID, s.class, s.z as redshift, s.zErr as redshiftError, s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_z, s.spectroSynFluxIvar_i, s.spectroSynFlux_i, s.spectroSynFlux_z, s.spectroSynFlux_u, s.spectroSynFlux_g, s.plate, s.mjd, s.fiberid, s.run2d, p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.fieldID, p.err_u, p.err_g, p.err_r, p.err_i, p.err_z, p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, p.petroRadErr_u, p.petroRadErr_g, p.petroRadErr_r, p.petroRadErr_z, p.petroRadErr_i
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   (s.class = 'STAR') AND (zWarning = 0)"""

def getObjTable(SQL_query, format="csv", fname="queryResults", TableName="",N=100000):
    """Retrieve table of N SDSS objects using SQL query
    params: SQL_query [str] - SQL query to be executed
            format [str] - format of table, options are 'csv', 'html', 'xml', 'json'
            TableName [str] - name of table, default is empty string
            N [int] - number of objects to retrieve, default is 100000, limit is 500000
    returns: query [str] - url of reformatted query, downloaded with wget and saved as {fname}.{format}
    """
    SQL_query = SQL_query.format(N)
    base_url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd="
    SQL_formatted = re.sub(r"\n", "%0D%0A", SQL_query)
    SQL_formatted = re.sub(r"\+", "%2B", SQL_formatted)
    SQL_formatted = re.sub(r"\s", "+", SQL_formatted)
    SQL_formatted = re.sub(r"=", "%3D", SQL_formatted)
    SQL_formatted = re.sub(r",", "%2C", SQL_formatted)
    SQL_formatted = re.sub(r"'", "%27", SQL_formatted)
    SQL_formatted = re.sub(r"\(", "%28", SQL_formatted)
    SQL_formatted = re.sub(r"\)", "%29", SQL_formatted)
    query = base_url + SQL_formatted + "%0D%0A%0D%0A&format={0}&TableName={1}".format(format, TableName)
    os.system("wget -O {0}.{1} '".format(fname,format) + query + "'")
    return query

def getSpectrum(id,format='png',spec='lite'):
    """Retrieve spectra of SDSS objects
    params: id [dict] - dictionary of object identifiers, requires at least 'specObjID'
            format [str] - format of spectrum, options are 'png', 'fits', 'csv'
            spec [str] - either 'lite' (default) or 'full'
    returns: url [str] - url of spectrum, downloaded with wget and saved as spectra/spectrum_{specObjID}.{format}
    """

    if format == 'png':
        base_url = "https://skyserver.sdss.org/dr18/en/get/specById.ashx?ID="
        url = base_url + str(int(id['specObjID']))
    elif format == 'fits' or format == 'csv':
        url = "http://dr18.sdss.org/optical/spectrum/view/data/format={4}/spec={5}?plateid={0}&mjd={1}&fiberid={2}&reduction2d={3}".format(str(int(id['plate'])), str(int(id['mjd'])), str(int(id['fiberid'])), str(int(id['run2d'])),format,spec)
    os.system("wget -O spectra/spectrum_{0}.{1} '".format(str(int(id['specObjID'])),format) + url + "'")
    return url

def getImg(id,opt="GL",width=512,height=512,scale=0.4):
    """Retrieve images of SDSS objects
    params: id [dict] - dictionary of object identifiers, requires at least 'specObjID', 'ra', and 'dec'
            opt [str] - options for image cutout, documentation: https://skyserver.sdss.org/dr14/en/help/docs/api.aspx#imgcutout; defaults to 'GL' (grid + label)
            width [int] - width of image in pixels
            height [int] - height of image in pixels
            scale [float] - scale of image in arcsec/pixel
    
    returns: url [str] - url of image cutout, downloaded with wget and saved as images/image_{specObjID}.jpeg 
    """

    url = "http://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?ra={0}&dec={1}&width={2}&height={3}&scale={4}&opt={5}".format(id['ra'],id['dec'],width,height,scale,opt)
    os.system("wget -O images/image_{0}.jpeg '".format(str(int(id['specObjID']))) + url + "'")
    return url

def getID(row,fname):
    row += 2 # skip table name + column names
    with open(fname,"r") as f:
        lines = f.readlines()
        d = lines[row].split(",")
        colNames = lines[1].split(",")
    return dict(zip(colNames,d))

def getDF(CSVList):
    """Retrieve pandas dataframe from list of CSV files
    params: CSVList [list] - list of CSV files
    returns: df [pandas dataframe] - dataframe of CSV files, assumes they share columnns
    """
    dfList = [pd.read_csv(CSVList[i],header=1) for i in range(len(CSVList))]
    df = pd.concat(dfList)
    return df

def clean(CSVList,save=True):
    print("getting combined dataframe")
    combined_df = getDF(CSVList)
    mask = (combined_df["u"] > 0) & (combined_df["g"] > 0) & (combined_df["r"] > 0) & (combined_df["i"] > 0) & (combined_df["z"] > 0) & (combined_df["err_u"] > 0) & (combined_df["err_g"] > 0) & (combined_df["err_r"] > 0) & (combined_df["err_i"] > 0) & (combined_df["err_z"] > 0) & (combined_df["redshiftError"] > 0)
    combined_df = combined_df[mask]
    df_noSize = combined_df.drop(columns=["petroRad_u","petroRad_g","petroRad_r","petroRad_i","petroRad_z","petroRadErr_u","petroRadErr_g","petroRadErr_r","petroRadErr_i","petroRadErr_z"]) #create a new dataframe without size information
    if save:
        df_noSize.to_csv("combined_noSize.csv",index=False) #save the resulting dataframe to a new CSV file
        print("saved combined_noSize.csv")
    else:
        print("obtained noSize dataframe, starting size cleaning")
    #clean size columns
    df_wSize = combined_df.copy() #create a copy of the dataframe to clean the size columns
    strLen = 0
    cols2Check = ["petroRadErr_u","petroRadErr_g","petroRadErr_r","petroRadErr_i","petroRadErr_z"] #need at least one to have an error
    mask = df_wSize[cols2Check] < 0
    errCounts = np.sum(mask,axis=1)
    inds2Check = np.where((errCounts <= 2) & (errCounts > 0))[0] #get the indices of the rows with size errors in less than 2 bands
    discard = np.where(errCounts > 2)[0] #throw away rows with errors in more than 2 bands
    for i in inds2Check: #note that this takes a long time to run...probably a more efficient way but only needs to be done once!
        mags = df_wSize.iloc[i][["u","g","r","i","z"]] #magnitudes in each band
        total = sum(mags)
        badBands = [j for j in range(len(cols2Check)) if mask.iloc[i][j]]
        badMags = [mags[j] for j in badBands]
        badTotal = sum(badMags)
        if badTotal/total < 0.1: #if the object is not bright in the bands with the size errors, replace the size errors with 0
            colNames = ["petroRad_u","petroRad_g","petroRad_r","petroRad_i","petroRad_z"]
            toReplace = [colNames[j] for j in badBands]
            df_wSize.iloc[i][[toReplace]] = 0.0
            toReplace = [cols2Check[j] for j in badBands]
            df_wSize.iloc[i][[toReplace]] = 0.0 #set the error also to "zero" to indicate that we replaced this size
        else: #otherwise throw out the row
            np.append(discard,i)
        strLen = trackPercent(i,df_wSize.shape[0],strLen)
    toDrop = df_wSize.iloc[discard]
    df_wSize.drop(toDrop.index,inplace=True)
    if save:
        df_wSize.to_csv("combined_wSize.csv",index=False) #save the resulting dataframe to a new CSV file
        print("saved combined_wSize.csv")
    else:
        return df_noSize,df_wSize

def visualizeDF(df,cols2Plot,x,nc=3):
    """Visualize dataframe as boxplot
    params: df [pandas dataframe] - dataframe to visualize
            cols2Plot [list] - list of columns to plot
            x - x variable for box plot
            nc [int] - number of columns to plot per row
    returns: fig,ax [matplotlib figure, axis] - figure and axis of plot
    """
    Nc = len(cols2Plot)
    nr = int(np.ceil(Nc/nc)) #number of rows needed to plot all data columns with nc columns per row

    fig, axs = plt.subplots(nr, nc, figsize=(16*(nc/nr),12*(nr/nc)),constrained_layout=True)

    for i, column in enumerate(cols2Plot):
        ax = axs.flatten()[i]
        try:
            sns.boxplot(x=x, y=df[column], ax=ax)
            ax.set_title(column)
            ax.set_ylabel(column)
            ax.set_xlabel('') 
        except:
            "error at column {0}".format(column)
    for i in range(nr*nc-Nc):
        fig.delaxes(axs.flatten()[-(1+i)]) #delete extra axes
    return fig,axs




