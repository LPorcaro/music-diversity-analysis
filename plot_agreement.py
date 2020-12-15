#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats.stats import pearsonr

INFILE = "data/Agree/cohen_kappa_20201211.csv"

plt.rcParams.update({'font.size': 16})


if __name__ == '__main__':
    
    df = pd.read_csv(INFILE)

    x = np.arange(df.shape[0])
    x_ticks = ["G{}".format(x) for x in np.arange(1, len(df)+1)]

    cs = ['b','r','g']
    fmts = ['o','^','x']
    TV_idx = ['TV', 'TV-A', 'TV-B']
    AD_idx = ['AD', 'AD-A', 'AD-B']
    MX_idx = ['MIX', 'MIX-A', 'MIX-B']
    TT = ['Track Variety Task', 'Artist Diversity Task', 'Mixed Task']
    LL = ['All Tasks', "Student Tasks", "Friends Tasks"]

    idxs = [TV_idx, AD_idx, MX_idx]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    plt.setp(axs, xticks=x, xticklabels=x_ticks)

    for p,idx in enumerate(idxs):
        axs[p].set_title(TT[p])
        for c,_ in enumerate(idx):

            # axs[p].plot(df[idx[c]].iloc[:-1],color=cs[c])

            # upline = df[idx[c]].iloc[-1]-0.05
            # downline = df[idx[c]].iloc[-1]+0.05
            # axs[p].axhline(y=df[idx[c]].iloc[-1], color=cs[c], linestyle='--')
            # axs[p].axhline(y=upline, color=cs[c], linestyle=':')
            # axs[p].axhline(y=downline, color=cs[c], linestyle=':')
            # axs[p].fill_between(x_axis, upline, downline, alpha=0.2)
            # axs[p].set_xlim([0,len(df)-2])
            # axs[p].set_ylim([0,1])

            kappas = [x.split() for x in df[idx[c]].values]
            y = []
            CI_lows = []
            CI_highs = []
            for el in kappas:
                if len(el) == 2:
                    k = float(el[0])
                    CI_low = k-eval(el[1])[0]
                    CI_high = eval(el[1])[1]-k
                else:
                    k = -1
                    CI_low = 0
                    CI_high = 0

                y.append(k)
                CI_lows.append(CI_low)
                CI_highs.append(CI_high)

                
            axs[p].errorbar(x[:-1], y[:-1], [CI_lows[:-1], CI_highs[:-1]], color=cs[c], linestyle='None', fmt=fmts[c])
            axs[p].axhline(y=y[-1], color=cs[c], linestyle='--', marker=fmts[c])
            axs[p].axhline(y=y[-1]+CI_highs[-1], color=cs[c], linestyle=':')
            axs[p].axhline(y=y[-1]-CI_lows[-1], color=cs[c], linestyle=':')
            axs[p].set_ylim([0,0.85])
            axs[p].set_xticklabels(x_ticks, rotation=45, ha='right')
            axs[p].grid(axis='x')


    patches = []
    for c, label in enumerate(LL):
        # patches.append(mpatches.Patch(color=cs[c], label=label))
        patches.append(mlines.Line2D([], [], color=cs[c], marker=fmts[c], markersize=10, label=label))

    patches.append(mlines.Line2D([], [], color='black', linestyle='--', label='All Participants'))
    patches.append(mlines.Line2D([], [], color='black', linestyle=':', label='CI 95%'))


    plt.legend(handles=patches)
    
    # fig.suptitle("Inter-rater reliability - average pairwise Cohen's {} coefficient".format(r'$\kappa$'), size=20)
    plt.show()  
