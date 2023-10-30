#!/usr/bin/env python
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import numpy as np
import sys,math
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def genDTDataMCMC(cleanFile="../MCMC/data/combined_cleaned.csv"):
    df = pd.read_csv(cleanFile)
    columns2ignore = ["rotation","singlePeak","doublePeak"]
    print("Reformatting input for DT")
    print("\nGenerating labels...\n")
    df["label"] = ['']*len(df)
    for i in range(len(df)):
        label = "singlePeak" if df["singlePeak"][i] == 1 else "doublePeak"
        df["label"][i] = label
        strLen = trackPercent(i,len(df),strLen)
    df.drop(columns=columns2ignore,inplace=True)
    print("\nSaving data...\n")
    df.to_csv("DTMCMCData.csv",index=False)

def genDTDataSDSS(cleanFile="../SDSS/combined_noSize_cleaned.csv"):
    df = pd.read_csv(cleanFile)
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
    print("Reformatting input for DT")
    df.drop(columns=columns2ignore,inplace=True)
    print("\nsaving data")
    out = "DTSDSS_noSizeData.csv" if "noSize" in cleanFile else "DTSDSS_wSizeData.csv"
    df.to_csv(out,index=False)

def genResults(DTfile = "DTMCMCData.csv",testSize=0.3,forest=False,maxDepth=3):
    df = pd.read_csv(DTfile)
    if "MCMC" in DTfile:
        X = df.drop(columns=["label"])
        Y = df["label"]
    else:
        X = df.drop(columns=["class"])
        Y = df["class"]
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=testSize)
    if forest:
        clf = RandomForestClassifier()
    else:
        clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrain,Ytrain)
    Ypred = clf.predict(Xtest)
    accuracy = accuracy_score(Ytest,Ypred)
    print("Accuracy: {:.2f}".format(accuracy))
    if forest:
        features = X.columns
        feature_importances = pd.Series(clf.feature_importances_,index=features)
        feature_importances_string = ""
        for feature in features:
            feature_importances_string += "{}: {:.2f}\n".format(feature,feature_importances[feature])
        print("Feature importances:\n"+feature_importances_string)
    else:
        names = [c for c in X.columns]
        export_graphviz(clf,out_file="tree.dot",feature_names=names,class_names=clf.classes_,filled=True,rounded=True,fontname="DejaVu Serif",max_depth=maxDepth)
        treeName = "DT_MCMC" if "MCMC" in DTfile else ("DT_SDSS_noSize" if "noSize" in DTfile else "DT_SDSS_wSize")
        graphviz.render("dot",filepath="tree.dot",format="png",outfile=treeName+".png") 
    return Ytest,Ypred,accuracy

def genPlot(Ytest,Ypred,accuracy):
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    ax.set_title("Accuracy: {:.2f}".format(accuracy))
    CM = confusion_matrix(Ytest,Ypred)
    clusters = np.unique(Ytest); nClusters = len(clusters)
    cmap = colormaps['inferno'].resampled(12)
    cmap = [cmap(i) for i in np.linspace(0,0.8,10)]
    cmap = ListedColormap(cmap)
    SM = ax.imshow(CM,cmap=cmap,vmin=0.0,vmax=np.max(CM))
    ax.set_xticks(np.arange(nClusters))
    ax.set_yticks(np.arange(nClusters))
    ax.set_xticklabels(["{}".format(i) for i in clusters])
    ax.set_yticklabels(["{}".format(i) for i in clusters])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right',size='5%',pad=0.0)
    cb = fig.colorbar(SM,cax=cax,fraction=0.046, pad=0.04, ticks=[0.1*i*np.max(CM) for i in range(11)])
    cb.ax.set_yticklabels(["{:.2g}".format(0.1*i) for i in range(11)])
    cb.set_label("Fraction of maximum counts")
    for ii in range(nClusters):
        for j in range(nClusters):
            ax.text(j,ii,CM[ii,j],ha="center",va="center",color="w")
    return fig,ax