#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from genNBData import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

font = {'family' : 'DejaVu Serif',
    'weight' : 'normal',
    'size'   : 16}
plt.rc('font', **font) #set all plot attribute defaults

def genResults(NBfile="NBMCMCData.csv",testSize=0.3,weighted=False):
    df = pd.read_csv(NBfile)
    if "MCMC" in NBfile:
        X = df.drop(columns=["label"])
        Y = df["label"]
    else:
        X = df.drop(columns=["class"])
        Y = df["class"]
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=testSize)
    clf = MultinomialNB(force_alpha=True)
    if weighted and "SDSS" in NBfile:
        cleanFile = "../SDSS/combined_noSize_cleaned.csv" if "noSize" in NBfile else "../SDSS/combined_wSize_cleaned.csv"
        weights = genSDSSweights(cleanFile=cleanFile)
        trainWeights = weights[Xtrain.index]
        clf.fit(Xtrain,Ytrain,sample_weight=trainWeights)
    else:
        clf.fit(Xtrain,Ytrain)

    Ypred = clf.predict(Xtest)
    accuracy = clf.score(Xtest,Ytest) if not weighted else clf.score(Xtest,Ytest,sample_weight=weights[Xtest.index])
    print("Accuracy: {:.2f}%".format(accuracy*100))
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