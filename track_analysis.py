#!/usr/bin/env python
# encoding: utf-8

import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import curve_fit
from scipy.spatial.distance import cosine, cityblock, euclidean
from scipy import stats
from matplotlib import cm

# HEADER = ['danceable', 'not_danceable',
#           # 'female', 'male',
#           # 'acoustic', 'not_acoustic',
#           'aggressive', 'not_aggressive',
#           # 'electronic', 'not_electronic',
#           'happy', 'not_happy',
#           'party', 'not_party', 
#           'relaxed', 'not_relaxed',
#           'sad','not_sad',
#           'bright', 'dark',
#           'atonal', 'tonal',
#           'instrumental', 'voice',
#           # 'ambient', 'dnb', 'house', 'techno', 'trance',
#           'Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5']



HEADER = ["silence_rate_30dB_mean", "silence_rate_30dB_std",
              "silence_rate_60dB_mean", "silence_rate_60dB_std",
              "dynamic_complexity",
              "dissonance_mean", "dissonance_std",
              "average_loudness",
              "pitch_salience_mean", "pitch_salience_std",
              "spectral_entropy_mean", "spectral_entropy_std",
              "zerocrossingrate_mean","zerocrossingrate_std",
              "hpcp_crest_mean", "hpcp_crest_std",
              "hpcp_entropy_mean","hpcp_entropy_std",
              "length",
              "onset_rate",
              "bpm"]

HIGH_LEVEL = ["danceability",
              # "gender",
              # "mood_acoustic",
              "mood_aggressive",
              # "mood_electronic",
              "mood_happy",
              "mood_party",
              "mood_relaxed",
              "mood_sad",
              "timbre",
              "tonal_atonal",
              "voice_instrumental",
              # "genre_electronic",
              "moods_mirex"]

RHYTHM_FEAT = ["onset_rate"]

LOW_LEVEL = ["silence_rate_30dB", "silence_rate_60dB",
             "dynamic_complexity", 
             "dissonance",
             "average_loudness","pitch_salience",
             "spectral_entropy","zerocrossingrate"]

TONAL = ["hpcp_crest", "hpcp_entropy"]

IN_FOLDER = "/home/lorenzo/Data/divsurvey/essentia_extractor_music/TV"

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)


 
if __name__ == '__main__':

    # Extract Features
    extract = False
    high_level = False

    if extract:
            with open("data/TV/TVFeatLow_out_20201103.csv", 'w+') as outf:
                _writer = csv.writer(outf)
                _writer.writerow(HEADER)
                for file in sorted(os.listdir(IN_FOLDER)):
                    print("Processing:",file)
                    infile = os.path.join(IN_FOLDER, file)

                    # Import json
                    with open(infile) as f:
                        d = json.load(f)

                    FeatDict = {}

                    if high_level:
                        for feat in HIGH_LEVEL:
                            FeatDict = {**FeatDict, **d["highlevel"][feat]["all"]}
                        # Write out
                        _writer.writerow([FeatDict[x] for x in FeatDict])
                    else:
                        for feat in LOW_LEVEL:
                            try:
                                FeatDict["_".join([feat,"mean"])] = d["lowlevel"][feat]["mean"]
                                FeatDict["_".join([feat,"std"])] = d["lowlevel"][feat]["stdev"]
                            except:
                                FeatDict[feat] = d['lowlevel'][feat]

                        for feat in TONAL:
                            FeatDict["_".join([feat,"mean"])] = d["tonal"][feat]["mean"]
                            FeatDict["_".join([feat,"std"])] = d["tonal"][feat]["stdev"]


                        FeatDict["length"] = d['metadata']["audio_properties"]["length"]
                        FeatDict["onset_rate"] = d['rhythm']['onset_rate'] 
                        FeatDict["bpm"] = d['rhythm']['bpm']


                        # Write out
                        _writer.writerow([FeatDict[x] for x in HEADER])



    # Compute cosine distance
    else:
        tracks_feat = []
        with open("data/TV/TVFeatHigh_out_20201103.csv", 'r') as inf:
            _reader = csv.reader(inf)
            header = next(_reader)
            for row in _reader:
                tracks_feat.append([float(x) for x in row])


        # Normalization
        tracks_feat = np.array(tracks_feat)
        tracks_feat = (tracks_feat - tracks_feat.min(0)) / tracks_feat.ptp(0)

        # Distance Matrix
        X = np.zeros((len(tracks_feat), len(tracks_feat)))
        distances = []
        for c1,t1 in enumerate(tracks_feat):
            for c2,t2 in enumerate(tracks_feat):
                if c2 > c1:
                    dist = cosine(t1,t2)
                    X[c1,c2] = X[c2,c1] = dist
                    distances.append(dist)

        out = X

        # ### Normality test ###
        stat, p = stats.shapiro(distances)
        print('Statistics={}, p={}'.format(stat, p))
        

        bin_heights, bin_borders, _ = plt.hist(distances, bins='auto', label='histogram')
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
        
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
        plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='fit')
        plt.legend()
        plt.show()


        ### Wilcoxon signed-rank test ### 
        dist_groups = []
        c = 0
        while c<32:
            T = np.triu(out[c:c+4, c:c+4])
            dist_groups.append(T[T>0])
            c+=4

        print(stats.wilcoxon(dist_groups[0], dist_groups[1]))
        print(stats.wilcoxon(dist_groups[2], dist_groups[3]))
        print(stats.wilcoxon(dist_groups[4], dist_groups[5]))
        print(stats.wilcoxon(dist_groups[6], dist_groups[7]))


        # ### Max and Min distances ###
        # min_v = out[np.where(out>0)].min()
        # max_v = out[np.where(out>0)].max()
        # print(min_v, max_v, np.where(out==min_v), np.where(out==max_v))

        # # Save Matrix and Plot it
        # np.savetxt("data/TV/TVFeatLow_CS_out_20201103.csv", out, delimiter=',', fmt='%.4f')


        # figure = plt.figure() 
        # axes = figure.add_subplot(111) 
        # caxes = axes.matshow(out, interpolation ='nearest', cmap=cm.Spectral_r) 
        # figure.colorbar(caxes) 
        # plt.show()


        # # Compute Stats for Group
        # c=0
        # while c<32:
        #     T = np.triu(out[c:c+4, c:c+4])
        #     # print(np.triu(out[c:c+4, c:c+4]))
        #     st = stats.describe(T[T>0.000000001])
        #     print("min-max:",st[1])
        #     print("avg:", st[2])

        #     v = []
        #     for i in range (T.shape[0]):
        #         for j in range(T.shape[1]):
        #             if j>i:
        #                 v.append(T[i,j])
        #     print("gmean:", stats.mstats.gmean(v))
        #     print()
        #     c+=4