#!/usr/bin/env python
# encoding: utf-8

import csv
from scipy.stats.stats import pearsonr
import pandas as pd
from matplotlib import pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D



INPUT_FILE = "data/Users/Users_20201201.csv"
INPUT_FILE = "data/TV/TVFeatAll_20201125.csv"


# G1 = G2 = ["FormalTraining","PlayiningTime","TasteVariety","ElectronicMusic","ElectronicTasteVariety","GAPScore"]

G1 = ["danceable","aggressive","happy","party","relaxed","sad","dark","atonal","voice"]

G2 = ["silence_rate_30dB_mean","silence_rate_30dB_std","silence_rate_60dB_mean","silence_rate_60dB_std","dynamic_complexity","dissonance_mean","dissonance_std","average_loudness","pitch_salience_mean","pitch_salience_std","spectral_entropy_mean","spectral_entropy_std","zerocrossingrate_mean","zerocrossingrate_std","hpcp_crest_mean","hpcp_crest_std","hpcp_entropy_mean","hpcp_entropy_std","onset_rate","bpm"]
# G1 = ["PlayiningTime", "TasteVariety", "ElectronicMusic", "ElectronicTasteVariety"]
# G2 = ["ScoreMainstream", "ScoreSurvey", "MeanScore", "SocialScore", "GAPScore"]
# G2 = ["GAPScore"]

if __name__ == '__main__':


    df = pd.read_csv(INPUT_FILE)
    # print(df)
    done = set()

    corrs = []
    for g1 in G1:
        for g2 in G2:
            if g1!=g2:
                if (g1,g2) not in done or (g2,g1) not in done:
                    ro,p = pearsonr(df[g1],df[g2])
                    corrs.append((g1,g2,ro, p))
                    done.add((g1,g2))
                    done.add((g2,g1))


    sorted_by_second = sorted(corrs, key=lambda tup: tup[2])

    for g in G1:
        for el in sorted_by_second:
            if g==el[0] and "silence_rate" not in el[1]:
                print(el)
                break
        




    # # Plot correlation EM ETV
    # g1 = df[(df["ElectronicMusic"]==5) & (df["ElectronicTasteVariety"]==5)].index
    # g2 = df[((df["ElectronicMusic"]==5) & (df["ElectronicTasteVariety"]==4)) | 
    #         ((df["ElectronicMusic"]==4) & (df["ElectronicTasteVariety"]==5)) ].index

    # g3 = df[(df["ElectronicMusic"]==4) & (df["ElectronicTasteVariety"]==4)].index
    # g4 = df[((df["ElectronicMusic"]==4) & (df["ElectronicTasteVariety"]==3)) | 
    #         ((df["ElectronicMusic"]==3) & (df["ElectronicTasteVariety"]==4)) ].index

    # g5 = df[(df["ElectronicMusic"]==3) & (df["ElectronicTasteVariety"]==3)].index
    # g6 = df[((df["ElectronicMusic"]==3) & (df["ElectronicTasteVariety"]==2)) | 
    #         ((df["ElectronicMusic"]==2) & (df["ElectronicTasteVariety"]==3)) ].index

    # g7 = df[(df["ElectronicMusic"]==2) & (df["ElectronicTasteVariety"]==2)].index
    # g8 = df[((df["ElectronicMusic"]==2) & (df["ElectronicTasteVariety"]==1)) | 
    #         ((df["ElectronicMusic"]==1) & (df["ElectronicTasteVariety"]==2)) ].index

    # g9 = df[(df["ElectronicMusic"]==1) & (df["ElectronicTasteVariety"]==1)].index


    # print(sum([len(x) for x in [g1,g2,g3,g4,g5,g6,g7,g8,g9]]))
    # # print(df.iloc[g2])


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')


    # xs = df["ElectronicMusic"]
    # ys = df["ElectronicTasteVariety"]
    # zs = df["NewScore"]

    # ax.scatter(xs=xs, ys=ys, zs=zs, zdir='z', s=40, c=zs, depthshade=True, cmap='Spectral')
    # # ax.set_xlabel("Electronic Music Listening")
    # ax.set_ylabel("Electronic Music Variety")
    # ax.set_zlabel("Social Score Questionnaire")

    # ax.xaxis.set_ticklabels([])
    # # plt.tick_params(left=False,
    # #                 # bottom=False,
    # #                 labelleft=False)
    # #                 # labelbottom=False)

    # plt.show()
