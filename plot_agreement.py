#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats.stats import pearsonr

INFILE = "data/Agree/cohen_kappa_20201013.csv"

if __name__ == '__main__':
    
    df = pd.read_csv(INFILE)

    x_axis = np.arange(0, len(df))
    x_ticks = np.arange(1, len(df)+1)

    cs = ['b','r','g']
    TV_idx = ['TV', 'TV-A', 'TV-B']
    AD_idx = ['AD', 'AD-A', 'AD-B']
    MX_idx = ['MIX', 'MIX-A', 'MIX-B']
    TT = ['Track Variety', 'Artist Diversity', 'Mixed']
    LL = ['All Tasks', "Student Tasks", "Friends Tasks"]

    idxs = [TV_idx, AD_idx, MX_idx]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    plt.setp(axs, xticks=x_axis, xticklabels=x_ticks)

    for p,idx in enumerate(idxs):
        axs[p].set_title(TT[p])
        for c,_ in enumerate(idx):

            axs[p].plot(df[idx[c]].iloc[:-1],color=cs[c])

            upline = df[idx[c]].iloc[-1]-0.05
            downline = df[idx[c]].iloc[-1]+0.05
            axs[p].axhline(y=df[idx[c]].iloc[-1], color=cs[c], linestyle='--')
            axs[p].axhline(y=upline, color=cs[c], linestyle=':')
            axs[p].axhline(y=downline, color=cs[c], linestyle=':')
            axs[p].fill_between(x_axis, upline, downline, alpha=0.2)
            axs[p].set_xlim([0,len(df)-2])
            axs[p].set_ylim([0,1])


    patches = []
    for c, label in enumerate(LL):
        patches.append(mpatches.Patch(color=cs[c], label=label))

    patches.append(mlines.Line2D([], [], color='black', linestyle='--', label='All Participants'))
    patches.append(mlines.Line2D([], [], color='black', linestyle=':', label='+/- 0.05'))

    plt.legend(handles=patches)
    fig.suptitle("Inter-rater Agreement - Average pairwise Cohen's kappa coefficient", size=20)
    plt.show()  