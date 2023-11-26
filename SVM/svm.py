#!/usr/bin/env python
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps
import numpy as np
from sklearn.metrics import confusion_matrix

font = {'family' : 'DejaVu Serif',
    'weight' : 'normal',
    'size'   : 16}
plt.rc('font', **font) #set all plot attribute defaults

def genResults(dataFile = "SVMMCMCData.csv",testSize=0.3,extraDrop=[],kernel="rbf",C=1.0,gamma="auto",cache_size=1000,degree=3):
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
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel,C=C,gamma=gamma,cache_size=cache_size,degree=degree))
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

def CParamSweep(fileList = ["SVMMCMCData.csv","SVMSDSS_wSizeData.csv","SVMSDSS_noSizeData.csv"],Clist = [0.1,0.5,1.,2.,3.,4.,5.],kernel="rbf",degree=3,gamma="auto",cache_size=1000,testSize=0.3,extraDrop=[]):
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    #note that this takes a few hours to run with default options on full datasets
    rows = []
    for file in fileList:
        print("Working on {}".format(file))
        for C in Clist:
            print("C = {}".format(C))
            trainTestSplit,Ypred,score = genResults(dataFile=file,C=C,kernel=kernel,degree=degree,gamma=gamma,cache_size=cache_size,testSize=testSize,extraDrop=extraDrop)
            rows.append([file,C,score])
            CMfig,CMax = genPlot(trainTestSplit[-1],Ypred,score,save="{}_kernel={}_C={}.png".format(file.split(".")[0],kernel,C))
    df = pd.DataFrame(rows,columns=["file","C","score"])
    ax = df.plot.line(x="C",y="score",marker="o")
    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig("CParamSweep.png",dpi=300)
    df.to_csv("CParamSweep.csv",index=False)
    return fig,ax,df

def kernelSweep(fileList = ["SVMMCMCData.csv","SVMSDSS_wSizeData.csv","SVMSDSS_noSizeData.csv"],kernels = ["rbf","linear","poly","sigmoid"],
                degree=3,gamma="auto",C=1.,cache_size=1000,testSize=0.3,extraDrop=[]):
    #note this takes a *long* time to run (roughly a day with default options, the sigmoid kernel is the worst offender)
    rows = []
    for file in fileList:
        print("Working on {}".format(file))
        for kernel in kernels:
            print("kernel = {}".format(kernel))
            trainTestSplit,Ypred,score = genResults(dataFile=file,kernel=kernel,degree=degree,gamma=gamma,C=C,cache_size=cache_size,testSize=testSize,extraDrop=extraDrop)
            rows.append([file,kernel,score])
            CMfig,CMax = genPlot(trainTestSplit[-1],Ypred,score,save="{}_kernel={}_C={}.png".format(file.split(".")[0],kernel,C))
    df = pd.DataFrame(rows,columns=["file","kernel","score"])
    ax = df.plot.bar(x="kernel",y="score",rot=0)
    ax.set_xlabel("kernel")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig = ax.get_figure()
    fig.savefig("kernelSweep.png",dpi=300)
    df.to_csv("kernelSweep.csv",index=False)
    return fig,ax,df

def groupedBarVis(df):
    fig,ax = plt.subplots(1,1,figsize=(14,10))
    xlabels = np.unique(df.kernel)
    x = np.arange(len(xlabels))
    width = 0.3
    for i in range(len(xlabels)):
        measurements = np.array(df[df.kernel == xlabels[i]].score*100)
        measurements = [measurements[0],measurements[2],measurements[1]] #swap no/with sizes to match line plot
        labels = ["DiskWind","SDSS (no sizes)","SDSS (with sizes)"] if i == 0 else None
        colors = ["dodgerblue","crimson","grey"]
        offsets = [width*i for i in range(len(measurements))]
        bars = ax.bar(x[i]+offsets,measurements,width,label=labels,color=colors,zorder=10)
        ax.bar_label(bars,padding=3,fmt="%.1f")
    l = ax.legend()
    l.set_title("Dataset")
    l.get_frame().set_edgecolor('none')
    ax.set_xticks(x+width,xlabels)
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Accuracy [% correct]")
    ax.tick_params(axis='y',which='minor',bottom=False)
    ax.set_ylim(0,105); ax.set_xlim(0-width,np.max(x)+3*width)
    ax.set_yticks(np.arange(0,110,10),["{}".format(i) for i in np.arange(0,110,10)],minor=False)
    ax.set_yticks(np.arange(0,100,2.5),["" for i in np.arange(0,100,2.5)],minor=True)
    ax.yaxis.grid(True,which='minor',color='k',alpha=0.05,zorder=0)
    ax.yaxis.grid(True,which='major',color='k',alpha=0.15,zorder=0)
    ax.set_title("SVM Accuracy (C=1.0, gamma=auto, degree=3)")
    return fig,ax

def redshiftRegression(dataFile="SVMSDSS_wSizeData.csv",testSize=0.3,extraDrop=["spectroSynFlux_i","spectroSynFlux_z","spectroSynFlux_u","spectroSynFlux_g"],C=1.0,maxIter=1000,tol=1e-5,eps=0.0):
    df = pd.read_csv(dataFile)
    cols2drop = ["class"] + ["redshift"] + extraDrop #extraDrop by default drops the spectroSynFlux columns
    X = df.drop(columns=cols2drop)
    Y = df["redshift"]
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=testSize)
    clf = make_pipeline(StandardScaler(), svm.LinearSVR(C=C,dual="auto",max_iter=maxIter,tol=tol,epsilon=eps)) #use LinearSVR as it is better suited for large datasets
    #note to use other kernels do svm.SVR(kernel=kernel,C=C,gamma=gamma,cache_size=cache_size,degree=degree,etc) -- this takes several hours to complete while the above runs in ~1 minute
    clf.fit(Xtrain,Ytrain)
    Ypred = clf.predict(Xtest)
    score = clf.score(Xtest,Ytest)
    return [Xtrain,Xtest,Ytrain,Ytest],Ypred,score