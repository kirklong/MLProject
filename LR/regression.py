#!/usr/bin/env python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps
import numpy as np
from sklearn.metrics import confusion_matrix

def genResults(dataFile = "../SVMMCMCData.csv",testSize=0.3,extraDrop=[],solver="saga",tol=1e-4,max_iter=1000,C=1.0):
    df = pd.read_csv(dataFile)
    if "MCMC" in dataFile:
        cols2drop = ["label"] + extraDrop
        X = df.drop(columns=cols2drop)
        Y = df["label"]
    else:
        cols2drop = ["class"] + extraDrop
        X = df.drop(columns=cols2drop)
        Y = df["class"]
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=testSize)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C,solver=solver,tol=tol,max_iter=max_iter))
    clf.fit(Xtrain,Ytrain)
    Ypred = clf.predict(Xtest)
    score = clf.score(Xtest,Ytest)
    return [Xtrain,Xtest,Ytrain,Ytest],Ypred,score

def genPlot(Ytest,Ypred,accuracy,save=False):
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
    if save != False:
        fig.savefig(save,dpi=300)
    return fig,ax