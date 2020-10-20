#!/usr/bin/env python
# encoding: utf-8

import numpy as np 
import gower
import pandas as pd 
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':

    # Import Data and Compute distance
    df = pd.read_csv("data/MIX/MixFeat_20201009_tra.csv")
    out = gower.gower_matrix(df)

    # Save Matrix and Plot it
    np.savetxt("data/MIX/Mix_GowerDistMatrix_20201009_tra.csv", out, delimiter=',', fmt='%.4f')
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(out, interpolation ='nearest', cmap=cm.Spectral_r) 
    figure.colorbar(caxes) 
    plt.show()

    print (np.max(out), np.min(out[out>0.000001]))

    # Compute Stats for Group
    c=0
    while c<32:
        T = np.triu(out[c:c+4, c:c+4])
        # print(np.triu(out[c:c+4, c:c+4]))
        st = stats.describe(T[T>0])
        print("min-max:",st[1])
        print("avg:", st[2])
        print("var:", st[3])
        print()
        c+=4
    



