#!/usr/bin/env python

#k-means clustering script
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler as sknormalize
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def kMeans(dataFile="MCMC_combined_cluster.csv",k=3,thin=False,thinFrac=0.1,score=True,normalize=False):
    df = pd.read_csv(dataFile,header=None)
    if thin:
        df = df.sample(frac=thinFrac)
    if normalize:
        scaler = sknormalize()
        df = pd.DataFrame(scaler.fit_transform(df.values),columns=df.columns)
    kmeans = KMeans(n_clusters=k,n_init="auto").fit(df.values)
    labels = kmeans.labels_
    if score:
        score = silhouette_score(df.values,labels)
        return kmeans,score
    else:
        return kmeans
    
def main(kRange=range(2,11),save=True,thin=False,thinFrac=0.1,combine=False,generate=False,normalize=False):
    dataFiles = ["MCMC_singlePeak_cluster.csv","MCMC_doublePeak_cluster.csv","SDSS_noSize_STAR_cluster.csv","SDSS_noSize_GALAXY_cluster.csv","SDSS_noSize_QSO_cluster.csv","SDSS_wSize_STAR_cluster.csv","SDSS_wSize_GALAXY_cluster.csv","SDSS_wSize_QSO_cluster.csv"]
    returnList = []
    if combine:
        if generate:
            print("Generating combined data files")
            MCMCdfSingle = pd.read_csv("MCMC_singlePeak_cluster.csv",header=None)
            MCMCdfDouble = pd.read_csv("MCMC_doublePeak_cluster.csv",header=None)
            MCMCdf = pd.concat([MCMCdfSingle,MCMCdfDouble])
            SDSS_noSize_STARdf = pd.read_csv("SDSS_noSize_STAR_cluster.csv",header=None)
            SDSS_noSize_GALAXYdf = pd.read_csv("SDSS_noSize_GALAXY_cluster.csv",header=None)
            SDSS_noSize_QSOdf = pd.read_csv("SDSS_noSize_QSO_cluster.csv",header=None)
            SDSS_noSize_df = pd.concat([SDSS_noSize_STARdf,SDSS_noSize_GALAXYdf,SDSS_noSize_QSOdf])
            SDSS_wSize_STARdf = pd.read_csv("SDSS_wSize_STAR_cluster.csv",header=None)
            SDSS_wSize_GALAXYdf = pd.read_csv("SDSS_wSize_GALAXY_cluster.csv",header=None)
            SDSS_wSize_QSOdf = pd.read_csv("SDSS_wSize_QSO_cluster.csv",header=None)
            SDSS_wSize_df = pd.concat([SDSS_wSize_STARdf,SDSS_wSize_GALAXYdf,SDSS_wSize_QSOdf])
            MCMCdf.to_csv("MCMC_combined_cluster.csv",index=False)
            SDSS_noSize_df.to_csv("SDSS_noSize_combined_cluster.csv",index=False)
            SDSS_wSize_df.to_csv("SDSS_wSize_combined_cluster.csv",index=False)

        dataFiles = ["MCMC_combined_cluster.csv","SDSS_noSize_combined_cluster.csv","SDSS_wSize_combined_cluster.csv"]

    for dataFile in dataFiles:
        s = time.time()
        print("working on {}".format(dataFile))
        scores = []; results = []
        for k in kRange:
            kmeans,score = kMeans(dataFile=dataFile,k=k,thin=thin,thinFrac=thinFrac,normalize=normalize)
            scores.append(score); results.append(kmeans)
            print(f"{dataFile} with {k} clusters has silhouette score of {score}")
        maxInd = np.argmax(scores)
        print(f"best score of {scores[maxInd]} found with {kRange[maxInd]} clusters")
        returnList.append((results[maxInd],scores[maxInd],kRange[maxInd]))
        f = time.time()
        print(f"took {f-s} seconds")
    
    MCMC_columns = ['i', 'r̄', 'Mfac', 'rFac', 'f1', 'f2', 'f3', 'f4', 'pa', 'Sα', 'phaseAmplitude', 't_char','rBLR', 'FHWM']
    SDSS_wSize_columns = ['redshift','spectroSynFlux_i','spectroSynFlux_z', 'spectroSynFlux_u', 'spectroSynFlux_g', 'ra', 'dec', 'u', 'g', 'r', 'i','z', 'petroRad_u', 'petroRad_g', 'petroRad_r', 'petroRad_i', 'petroRad_z']
    SDSS_noSize_columns = SDSS_wSize_columns[:-5]

    returnDictList = []
    for (i,dataFile) in enumerate(dataFiles):
        score = returnList[i][1]
        k = returnList[i][2]
        print("generating final clustering result (no thinning) for {0} with k = {1} and score = {2}".format(dataFile,k,score))
        kmeans = kMeans(dataFile=dataFile,k=k,thin=False,thinFrac=1,score=False)
        if "MCMC" in dataFile:
            columns = MCMC_columns
        elif "SDSS_wSize" in dataFile:
            columns = SDSS_wSize_columns
        elif "SDSS_noSize" in dataFile:
            columns = SDSS_noSize_columns
        else:
            print("ERROR: dataFile not recognized")
            return 1
        clusters = kmeans.cluster_centers_ #shape (k,len(columns))
        labels = kmeans.labels_ 
        returnDict = dict(labels=labels,clusters=clusters,columns=columns)
        returnDictList.append(returnDict)
        
    if save:
        print("saving results to kMeansResults.pkl")
        with open("kMeansResults.pkl","wb") as f:
            pickle.dump(returnDictList,f)
    
    return returnDictList

def classificationReport(saveFile="kMeansResults.pkl"):
    with open(saveFile,"rb") as f:
        results = pickle.load(f)
    combined = False
    if len(results) == 3:
        combined = True
        dataFiles = ["MCMC_combined_cluster.csv","SDSS_noSize_combined_cluster.csv","SDSS_wSize_combined_cluster.csv"]
    else:
        combined = False
        dataFiles = ["MCMC_singlePeak_cluster.csv","MCMC_doublePeak_cluster.csv","SDSS_noSize_STAR_cluster.csv","SDSS_noSize_GALAXY_cluster.csv","SDSS_noSize_QSO_cluster.csv","SDSS_wSize_STAR_cluster.csv","SDSS_wSize_GALAXY_cluster.csv","SDSS_wSize_QSO_cluster.csv"]

    for (i,result) in enumerate(results):
        labels = result["labels"]
        columns = result["columns"]
        df = pd.read_csv(dataFiles[i],header=None)
        df.columns = columns
        df["labels"] = labels
        if combined:
            if "MCMC" in dataFiles[i]:
                with open("MCMC_singlePeak_cluster.csv","r") as f:
                    lines = f.readlines()
                singlePeakN = len(lines)
                singlePeakList = ["singlePeak" for i in range(singlePeakN)]
                doublePeakList = ["doublePeak" for i in range(len(df)-singlePeakN)]
                df["class"] = singlePeakList + doublePeakList
            elif "SDSS_noSize" in dataFiles[i]:
                with open("SDSS_noSize_STAR_cluster.csv","r") as f:
                    lines = f.readlines()
                STAR_N = len(lines)
                with open("SDSS_noSize_GALAXY_cluster.csv","r") as f:
                    lines = f.readlines()
                GALAXY_N = len(lines)
                QSO_N = len(df) - STAR_N - GALAXY_N
                STAR_list = ["STAR" for i in range(STAR_N)]
                GALAXY_list = ["GALAXY" for i in range(GALAXY_N)]
                QSO_list = ["QSO" for i in range(QSO_N)]
                df["class"] = STAR_list + GALAXY_list + QSO_list
            elif "SDSS_wSize" in dataFiles[i]:
                with open("SDSS_wSize_STAR_cluster.csv","r") as f:
                    lines = f.readlines()
                STAR_N = len(lines)
                with open("SDSS_wSize_GALAXY_cluster.csv","r") as f:
                    lines = f.readlines()
                GALAXY_N = len(lines)
                QSO_N = len(df) - STAR_N - GALAXY_N
                STAR_list = ["STAR" for i in range(STAR_N)]
                GALAXY_list = ["GALAXY" for i in range(GALAXY_N)]
                QSO_list = ["QSO" for i in range(QSO_N)]
                df["class"] = STAR_list + GALAXY_list + QSO_list
            fOut = dataFiles[i][:-4]+"_resultWithClassifications.csv"
            df.to_csv(fOut,index=False)
        else:
            fOut = dataFiles[i][:-4]+"_result.csv"
            df.to_csv(fOut,index=False)
    return 0
    
def clusterClasses(): #checking if clustering can find QSO vs STAR vs GALAXY etc.
    MCMC_class_df = pd.read_csv("MCMC_combined_cluster_resultWithClassifications.csv")
    SDSS_noSize_class_df = pd.read_csv("SDSS_noSize_combined_cluster_resultWithClassifications.csv")
    SDSS_wSize_class_df = pd.read_csv("SDSS_wSize_combined_cluster_resultWithClassifications.csv")

    with open("kMeansResults.pkl","rb") as f:
        results = pickle.load(f)
    
    MCMC_labels = results[0]["labels"]
    SDSS_noSize_labels = results[1]["labels"]
    SDSS_wSize_labels = results[2]["labels"]

    kMaxMCMC = np.max(MCMC_labels)
    kMaxSDSS_noSize = np.max(SDSS_noSize_labels)
    kMaxSDSS_wSize = np.max(SDSS_wSize_labels)

    MCMC_singleExpected = np.sum(MCMC_class_df["class"] == "singlePeak")/len(MCMC_class_df)
    MCMC_doubleExpected = 1 - MCMC_singleExpected
    SDSS_noSize_STARExpected = np.sum(SDSS_noSize_class_df["class"] == "STAR")/len(SDSS_noSize_class_df)
    SDSS_noSize_GALAXYExpected = np.sum(SDSS_noSize_class_df["class"] == "GALAXY")/len(SDSS_noSize_class_df)
    SDSS_noSize_QSOExpected = np.sum(SDSS_noSize_class_df["class"] == "QSO")/len(SDSS_noSize_class_df)
    SDSS_wSize_STARExpected = np.sum(SDSS_wSize_class_df["class"] == "STAR")/len(SDSS_wSize_class_df)
    SDSS_wSize_GALAXYExpected = np.sum(SDSS_wSize_class_df["class"] == "GALAXY")/len(SDSS_wSize_class_df)
    SDSS_wSize_QSOExpected = np.sum(SDSS_wSize_class_df["class"] == "QSO")/len(SDSS_wSize_class_df)

    returnDict = []
    for k in range(kMaxMCMC+1):
        c = MCMC_labels == k
        singleFrac = np.sum(MCMC_class_df["class"][c] == "singlePeak")/np.sum(c)
        doubleFrac = 1 - singleFrac
        d = dict(dataset="MCMC", cluster=k,single=(singleFrac-MCMC_singleExpected)/MCMC_singleExpected,double=(doubleFrac-MCMC_doubleExpected)/MCMC_doubleExpected)
        returnDict.append(d)
    for k in range(kMaxSDSS_noSize+1):
        c = SDSS_noSize_labels == k
        STARfrac = np.sum(SDSS_noSize_class_df["class"][c] == "STAR")/np.sum(c)
        GALAXYfrac = np.sum(SDSS_noSize_class_df["class"][c] == "GALAXY")/np.sum(c)
        QSOfrac = np.sum(SDSS_noSize_class_df["class"][c] == "QSO")/np.sum(c)
        d = dict(dataset="SDSS_noSize", cluster=k,STAR=(STARfrac-SDSS_noSize_STARExpected)/SDSS_noSize_STARExpected,GALAXY=(GALAXYfrac-SDSS_noSize_GALAXYExpected)/SDSS_noSize_GALAXYExpected,QSO=(QSOfrac-SDSS_noSize_QSOExpected)/SDSS_noSize_QSOExpected)
        returnDict.append(d)
    for k in range(kMaxSDSS_wSize+1):
        c = SDSS_wSize_labels == k
        STARfrac = np.sum(SDSS_wSize_class_df["class"][c] == "STAR")/np.sum(c)
        GALAXYfrac = np.sum(SDSS_wSize_class_df["class"][c] == "GALAXY")/np.sum(c)
        QSOfrac = np.sum(SDSS_wSize_class_df["class"][c] == "QSO")/np.sum(c)
        d = dict(dataset="SDSS_wSize", cluster=k,STAR=(STARfrac-SDSS_wSize_STARExpected)/SDSS_wSize_STARExpected,GALAXY=(GALAXYfrac-SDSS_wSize_GALAXYExpected)/SDSS_wSize_GALAXYExpected,QSO=(QSOfrac-SDSS_wSize_QSOExpected)/SDSS_wSize_QSOExpected)
        returnDict.append(d)
    return returnDict

def reportClusterMeans(result): #where result is a dictionary, i.e. results[0] from kMeansResults.pkl
    for (i,label) in enumerate(result["columns"]):
        for k in range(0,4):
            print("{0} cluster {1} center: {2:.4g}".format(label,k+1,result["clusters"][k,i]))


def visualizeClusterClasses():
    dictList = clusterClasses()
    xlabels = ["Disk-wind [1]","Disk-wind [2]","SDSS (no size) [1]","SDSS (no size) [2]","SDSS (with size) [1]","SDSS (with size) [2]","SDSS (with size) [3]","SDSS (with size) [4]"]
    fig,ax = plt.subplots(figsize=(17,5.5))
    x = np.arange(len(dictList)); width = 0.25; multiplier = 0
    for i in range(len(dictList)):
        d = dictList[i]
        if "MCMC" in d["dataset"]:
            measurements = np.array([d["single"],d["double"]])*100
            labels = ["single peak","double peak"] if i == 0 else None
            colors = ["dodgerblue","crimson"]
        else:
            measurements = np.array([d["STAR"],d["GALAXY"],d["QSO"]])*100
            labels = ["star","galaxy","quasar"] if i == 2 else None
            colors = ["goldenrod","cyan","orange"]
            width = 0.25
        
        offsets = [width*i for i in range(len(measurements))]
        bars = ax.bar(x[i]+offsets,measurements,width,label=labels,color=colors)
        ax.bar_label(bars,padding=3,fmt='%.1f')
    l = ax.legend(loc='upper left')
    ax.set_xticks(x+width,xlabels)
    ax.set_ylabel("Fractional difference from expected [%]")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis='y',which='minor',bottom=False)
    return fig,ax