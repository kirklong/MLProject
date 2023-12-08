#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Serif',
    'weight' : 'normal',
    'size'   : 16}
plt.rc('font', **font) #set all plot attribute defaults


df = pd.read_csv("FinalExamFakeData.csv")
cc = lambda score: np.max([0, 500*score - 1500])
amounts = [cc(score) for score in df['Credit worthiness']]
approved = [amount > 0 for amount in amounts]
df['credit amount'] = amounts
df['credit approved'] = approved

# 1. Plot the credit amount vs. the credit worthiness score
fig,ax = plt.subplots(figsize=(8,6))
nPerScore = np.array([sum(df['Credit worthiness']==score) for score in np.unique(df["Credit worthiness"])])
ax.plot(np.unique(df['Credit worthiness']),[cc(score) for score in np.unique(df["Credit worthiness"])],color='dodgerblue',ls='--',lw=2)
ax.scatter(np.unique(df['Credit worthiness']),[cc(score) for score in np.unique(df["Credit worthiness"])],color='dodgerblue',s=nPerScore*100)
ax.set_xlabel('Credit worthiness score (0-10 with 10 being best)')
ax.set_ylabel('Credit amount (dollars)')
ax.annotate("Fraction approved = {:.2f}%".format(np.sum(df['credit approved'])/len(df['credit approved'])*100),xy=(0.05,0.95),xycoords='axes fraction',ha='left',va='top')
ax.annotate("Average credit amount = ${:.2f}".format(np.mean(df['credit amount'])),xy=(0.05,0.90),xycoords='axes fraction',ha='left',va='top')
ax.annotate("Total credit offered = ${:.2f}".format(np.sum(df['credit amount'])),xy=(0.05,0.85),xycoords='axes fraction',ha='left',va='top')
ax.set_title("Credit amount vs. credit worthiness score")
ax.minorticks_on()
ax.grid(which='major',linestyle='-',color='black',alpha=0.3)
ax.grid(which='minor',linestyle=':',color='black',alpha=0.1)
fig.tight_layout()
fig.savefig("credit_amount_vs_credit_worthiness.png",dpi=300)
plt.close('all')


# 2. show percentage of marketable users based on top 14 rules
lhs_search = ["tv","switch","soundbar","controller","PS5","Xbox"]
rhs_recommend = ["soundbar","controller","tv","switch","game","game"]
#rules obtained from R analysis
recommendations = []
for i in range(len(df)):
    recommendation = []
    if df['Most recent search'][i] != df['Most recent purchase'][i]:
        recommendation.append(df['Most recent search'][i])
    for j in range(len(lhs_search)):
        if lhs_search[j] == df['Most recent search'][i] and rhs_recommend[j] != df['Most recent purchase'][i]:
            recommendation.append(rhs_recommend[j])
    recommendations.append(recommendation)
df['recommendations'] = recommendations



