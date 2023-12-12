#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import numpy as np
import pandas as pd
import os 
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

AUTOTUNE = tf.data.AUTOTUNE

def imgClassifier(dataFile="../SDSS/combined_wSize_cleaned.csv"):
    data = pd.read_csv(dataFile)
    files = os.listdir("images")

    ids = [int(f.split("_")[1].split(".")[0]) for f in files]
    labels = data[data['specObjID'].isin(ids)]['class'].values
    labels = [0 if l == "QSO" else 1 if l == "GALAXY" else 2 for l in labels]
    train = tf.keras.utils.image_dataset_from_directory(
        "images",
        subset = "training",
        labels = labels,
        image_size=(512,512),
        batch_size=32,
        validation_split=0.2,
        seed=42)
    val = tf.keras.utils.image_dataset_from_directory(
        "images",
        subset = "validation",
        labels = labels,
        image_size=(512,512),
        batch_size=32,
        validation_split=0.2,
        seed=42)
    norm_layer = tf.keras.layers.Rescaling(1./255)
    train = train.map(lambda x,y: (norm_layer(x),y))
    val = val.map(lambda x,y: (norm_layer(x),y))
    train = train.cache().prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    m = model.fit(train,validation_data=val,epochs=10)
    #35% accuracy...not better than guessing
    #need to make image cutouts smaller? hand select "good" images?
    return m

def redshiftRegression(dataFile="../SVM/SVMSDSS_wSizeData.csv",testSize=0.3,valSize=0.2,frac=1.,extraDrop=["spectroSynFlux_i","spectroSynFlux_z","spectroSynFlux_u","spectroSynFlux_g"],epochs=100,verbose=0,nPerLayer=4,nLayers=2,weightPow=2):
    X = pd.read_csv(dataFile).sample(frac=frac)
    cols2drop = extraDrop
    X.drop(columns=cols2drop,inplace=True)
    Y = X.pop("redshift")
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=testSize)
    Xtrain,Xval,Ytrain,Yval = train_test_split(Xtrain,Ytrain,test_size=valSize) #valSize*100% of training data for validation

    train_class = Xtrain.pop("class"); test_class = Xtest.pop("class"); val_class = Xval.pop("class") #remove class column from training, test, and validation data

    normalizer = layers.Normalization() #our dataframe has shape (n, features) so we want to normalize along the features axis (take mean and std of each feature)
    normalizer.adapt(np.array(Xtrain)) #calculate mean and std of redshifts in training data
    #deNorm = layers.Normalization(invert=True) #create denormalizer layer to invert normalization
    #deNorm.adapt(Xtrain) #adapt denormalizer layer to training data
    trainingWeights = np.power(Ytrain,weightPow) #weights for training data, higher redshifts are weighted more

    model = tf.keras.Sequential([layers.Input(shape=(10,)),normalizer]) 
    for i in range(nLayers):
        model.add(layers.Dense(nPerLayer,activation='relu'))
    model.add(layers.Dense(1)) #add output layer

    model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001)) #compile model with mean absolute error loss and adam optimizer

    history = model.fit(Xtrain,Ytrain,verbose=verbose,epochs=epochs,validation_data=(Xval,Yval),sample_weight=trainingWeights) #fit model to data
    Ypred = model.predict(Xtest).flatten() #predict on test data
    err = np.mean(np.abs(Ypred-Ytest)) 
    QSO_err = np.mean(np.abs(Ypred[test_class == "QSO"]-Ytest[test_class == "QSO"]))
    GAL_err = np.mean(np.abs(Ypred[test_class == "GALAXY"]-Ytest[test_class == "GALAXY"]))
    STAR_err = np.mean(np.abs(Ypred[test_class == "STAR"]-Ytest[test_class == "STAR"]))
    return [Xtrain,Xtest,Xval,Yval,Ytrain,Ytest],[train_class,val_class,test_class],[model,history,Ypred],[err,QSO_err,GAL_err,STAR_err]

def paramSweep(nLayers=[1,2,3,4,5],nPerLayer=[8,16,32,64,128,256],weightPow=[0,1,2],epochs=25,frac=0.5,verbose=0):
    results = pd.DataFrame(columns=["nLayers","nPerLayer","weightPow","history","avg_err","QSO_err","GAL_err","STAR_err"])
    for nL in nLayers:
        for nPL in nPerLayer:
            for w in weightPow:
                print("nLayers = {}, nPerLayer = {}, weightPow = {}".format(nL,nPL,w))
                res = redshiftRegression(nLayers=nL,nPerLayer=nPL,weightPow=w,epochs=epochs,frac=frac,verbose=verbose)
                h = res[1][1].history
                tmp = pd.DataFrame([[nL,nPL,w,h,res[2][0],res[2][1],res[2][2],res[2][3]]],columns=["nLayers","nPerLayer","weightPow","history","avg_err","QSO_err","GAL_err","STAR_err"])
                results = pd.concat([results,tmp],ignore_index=True)
    return results

def visualizeRes():
    res = pd.read_csv("sweep_results.csv")
    avg_err_layer = res.groupby("nLayers")[["avg_err","QSO_err","STAR_err","GAL_err"]].mean()
    avg_err_pow = res.groupby("weightPow")[["avg_err","QSO_err","STAR_err","GAL_err"]].mean()
    avg_err_nPerLayer = res.groupby("nPerLayer")[["avg_err","QSO_err","STAR_err","GAL_err"]].mean()
    avg_err_layer_pow0 = res[res["weightPow"] == 0].groupby("nLayers")[["avg_err","QSO_err","STAR_err","GAL_err"]].mean()
    avg_err_nPerLayer_pow0 = res[res["weightPow"] == 0].groupby("nPerLayer")[["avg_err","QSO_err","STAR_err","GAL_err"]].mean()

    #accuracy of QSO and GAL best fit are ~83% (mean of error / mean of redshift for that class from data) (get means by doing data.groupby("class")["redshift"].mean())
    best_overall = res.iloc[res["avg_err"].idxmin()]
    best_QSO = res.iloc[res["QSO_err"].idxmin()]
    best_STAR = res.iloc[res["STAR_err"].idxmin()]
    best_GAL = res.iloc[res["GAL_err"].idxmin()]

    #plot err as function of sweep parameters
    fig,ax = plt.subplots(1,3,figsize=(21,7),gridspec_kw={'wspace':0},sharey=True)
    ax[0].plot(avg_err_layer.index,avg_err_layer["avg_err"],label="Average error",color="black",lw=2,ls="--",marker="o",markersize=10)
    ax[0].plot(avg_err_layer.index,avg_err_layer["QSO_err"],label="QSO error",color="crimson",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[0].plot(avg_err_layer.index,avg_err_layer["GAL_err"],label="GAL error",color="dodgerblue",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[0].plot(avg_err_layer.index,avg_err_layer["STAR_err"],label="STAR error",color="goldenrod",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[0].set_xlabel("Number of hidden layers")
    ax[0].set_ylabel("Average absolute redshift error")

    ax[2].plot(avg_err_pow.index,avg_err_pow["avg_err"],label="Average error",color="black",lw=2,ls="--",marker="o",markersize=10)
    ax[2].plot(avg_err_pow.index,avg_err_pow["QSO_err"],label="QSO error",color="crimson",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[2].plot(avg_err_pow.index,avg_err_pow["GAL_err"],label="GAL error",color="dodgerblue",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[2].plot(avg_err_pow.index,avg_err_pow["STAR_err"],label="STAR error",color="goldenrod",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[2].set_xlabel("high redshift weight power")
    ax[2].set_xticks([0,1,2])

    ax[1].plot(avg_err_nPerLayer.index,avg_err_nPerLayer["avg_err"],label="Average error",color="black",lw=2,ls="--",marker="o",markersize=10)
    ax[1].plot(avg_err_nPerLayer.index,avg_err_nPerLayer["QSO_err"],label="QSO error",color="crimson",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[1].plot(avg_err_nPerLayer.index,avg_err_nPerLayer["GAL_err"],label="GAL error",color="dodgerblue",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[1].plot(avg_err_nPerLayer.index,avg_err_nPerLayer["STAR_err"],label="STAR error",color="goldenrod",lw=2,alpha=0.5,marker="o",markersize=10)
    ax[1].set_xlabel("Number of neurons per hidden layer")
    ax[1].set_xticks([8,16,32,64,128,256])
    l = ax[1].legend(loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.1))
    l.get_frame().set_edgecolor('none')

    for i in range(3):
        ax[i].minorticks_on()
        ax[i].grid(True,which='major',alpha=0.3)
        ax[i].grid(True,which='minor',alpha=0.1)
        ax[i].tick_params(axis='both',which='both',direction='in',top=True,right=True)
    return fig,ax



def predicted_vs_data(trainValTestSplit,labels,modelHistoryPred,errors):
    fig,ax = plt.subplots(figsize=(10,10))
    Xtrain,Xtest,Xval,Yval,Ytrain,Ytest = trainValTestSplit
    train_class,val_class,test_class = labels
    model,history,Ypred = modelHistoryPred
    err,QSO_err,GAL_err,STAR_err = errors
    ax.scatter(Ytest[test_class == "QSO"],Ypred[test_class == "QSO"],label="QSO (average error = {:.3g})".format(QSO_err),color="crimson",alpha=0.5)
    ax.scatter(Ytest[test_class == "GALAXY"],Ypred[test_class == "GALAXY"],label="GALAXY (average error = {:.3g})".format(GAL_err),color="dodgerblue",alpha=0.5)
    ax.scatter(Ytest[test_class == "STAR"],Ypred[test_class == "STAR"],label="STAR (average error = {:.3g})".format(STAR_err),color="goldenrod",alpha=0.5)
    ax.plot(Ytest,Ytest,label="Perfect fit",color="black",lw=2,ls="--")
    ax.set_xlabel("True redshift")
    ax.set_ylabel("Predicted redshift")
    l = ax.legend(title="Class (average error = {:.3g})".format(err),loc="upper left")
    l.get_frame().set_edgecolor('black')
    ax.minorticks_on()
    ax.grid(True,which='major',alpha=0.3)
    ax.grid(True,which='minor',alpha=0.1)
    ax.set_xlim(np.min(Ytest),np.max(Ytest)); ax.set_ylim(np.min(Ytest),np.max(Ytest))
    return fig,ax

def param_redshift_predictions(trainValTestSplit,labels,modelHistoryPred,errors):
    Xtrain,Xtest,Xval,Yval,Ytrain,Ytest = trainValTestSplit
    train_class,val_class,test_class = labels
    model,history,Ypred = modelHistoryPred
    err,QSO_err,GAL_err,STAR_err = errors
    features = Xtrain.columns; n_features = len(features)
    fig,axList = plt.subplots(2,5,figsize=(20,12),gridspec_kw={'wspace':0,'hspace':0},sharey=True)
    axList = axList.flatten()
    for i in range(n_features):
        ax = axList[i]
        ax.scatter(Xtest[test_class == "QSO"].iloc[:,i],Ytest[test_class == "QSO"],label="QSO (average error = {:.3g})".format(QSO_err),color="crimson",alpha=0.5,s=4)
        ax.scatter(Xtest[test_class == "GALAXY"].iloc[:,i],Ytest[test_class == "GALAXY"],label="GALAXY (average error = {:.3g})".format(GAL_err),color="dodgerblue",alpha=0.5,s=4)
        ax.scatter(Xtest[test_class == "STAR"].iloc[:,i],Ytest[test_class == "STAR"],label="STAR (average error = {:.3g})".format(STAR_err),color="goldenrod",alpha=0.5,s=4)
        ax.scatter(Xtest.iloc[:,i],Ypred,label="Predictions",color="black",s=2,alpha=0.2)
        if i == 0:
            ax.set_ylabel("True redshift",loc="center")
            ax.yaxis.set_label_coords(-0.1,0.0)
        if i >= 5:
            ax.set_xlabel(features[i])
        else:
            ax.xaxis.set_label_position('top')
            ax.xaxis.set_ticks_position('top')
            ax.set_xlabel(features[i])
        if i == 5:
            ax.set_yticks([i for i in range(0,7)])
        ax.minorticks_on()
        ax.grid(True,which='major',alpha=0.3)
        ax.grid(True,which='minor',alpha=0.1)
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax.set_ylim(np.min(Ytest),np.max(Ytest))

        if i==2:
            l = ax.legend(title="Class (average error = {:.3g})".format(err),loc="upper center",bbox_to_anchor=(0.5,1.3),ncol=4)
            l.get_frame().set_edgecolor('none')
    return fig,axList





