#!/usr/bin/env python
# encoding: utf-8

import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy import stats
from matplotlib import cm

HEADER = ['danceable', 'not_danceable',
          'female', 'male',
          'acoustic', 'not_acoustic',
          'aggressive', 'not_aggressive',
          'electronic', 'not_electronic',
          'happy', 'not_happy',
          'party', 'not_party', 
          'relaxed', 'not_relaxed',
          'sad','not_sad',
          'bright', 'dark',
          'atonal', 'tonal',
          'instrumental', 'voice',
          'ambient', 'dnb', 'house', 'techno', 'trance',
          'Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5',
          'bpm']


HIGH_LEVEL = ["danceability",
              "gender",
              "mood_acoustic",
              "mood_aggressive",
              "mood_electronic",
              "mood_happy",
              "mood_party",
              "mood_relaxed",
              "mood_sad",
              "timbre",
              "tonal_atonal",
              "voice_instrumental",
              "genre_electronic",
              "moods_mirex"]

IN_FOLDER = "/home/lorenzo/Data/divsurvey/essentia_extractor_music/MIX"

    
if __name__ == '__main__':

    # Extract Features
    extract = False
    if extract:
        with open("data/MIX/MixFeat_out_20201009.csv", 'w+') as outf:
            _writer = csv.writer(outf)
            _writer.writerow(HEADER)

            for file in sorted(os.listdir(IN_FOLDER)):
                print("Processing:",file)
                infile = os.path.join(IN_FOLDER, file)

                # Import json
                with open(infile) as f:
                    d = json.load(f)
                # Select Features
                FeatDict = {}
                for feat in HIGH_LEVEL:
                    FeatDict = {**FeatDict, **d["highlevel"][feat]["all"]}
                # FeatDict["bpm"] = (d['rhythm']['bpm'] -75) / (125)
                FeatDict["bpm"] = d['rhythm']['bpm']
                # Write out
                _writer.writerow([FeatDict[x] for x in FeatDict])


    # Compute cosine distance
    else:
        tracks_feat = []
        with open("data/MIX/MixFeat_out_20201009.csv", 'r') as inf:
            _reader = csv.reader(inf)
            header = next(_reader)
            for row in _reader:
                tracks_feat.append([float(x) for x in row])


        # Distance Matrix
        X = np.zeros((len(tracks_feat), len(tracks_feat)))
        for c1,t1 in enumerate(tracks_feat):
            for c2,t2 in enumerate(tracks_feat):
                    X[c1,c2] = cosine(t1,t2)


        # out = 1 - cosine_similarity(tracks_feat)
        out = X

        print (np.max(out), np.min(out[out>0.00001]))
        # Save Matrix and Plot it
        np.savetxt("data/MIX/CosineSim_MixFeat_out_20201009.csv", out, delimiter=',', fmt='%.4f')


        figure = plt.figure() 
        axes = figure.add_subplot(111) 
        caxes = axes.matshow(out, interpolation ='nearest', cmap=cm.Spectral_r) 
        figure.colorbar(caxes) 
        plt.show()


        # Compute Stats for Group
        c=0
        while c<32:
            T = np.triu(out[c:c+4, c:c+4])
            # print(np.triu(out[c:c+4, c:c+4]))
            st = stats.describe(T[T>0.000001])
            print("min-max:",st[1])
            print("avg:", st[2])
            print("var:", st[3])
            print()
            c+=4