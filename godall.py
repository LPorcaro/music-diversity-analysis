#!/usr/bin/env python
# encoding: utf-8


""" REFERENCE:
Boriah, S., Chandola, V., & Kumar, V. (2008). Similarity measures for
categorical data: A comparative evaluation. Society for Industrial and
Applied Mathematics - 8th SIAM International Conference on Data Mining 2008,
Proceedings in Applied Mathematics 130, 1, 243–254.
https://doi.org/10.1137/1.9781611972788.22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from collections import Counter
from itertools import combinations
from scipy import stats


INPUT = "data/AD/ArtistsFeatures_20201105.csv"


if __name__ == '__main__':

    df = pd.read_csv(INPUT)
    N = df.shape[0]

    # Compute the frequency of each attribute's values
    DictFreqAttr = {}
    for col in df.columns:
        DictFreqAttr[col] = Counter(df[col])

    # Compute the probability of each attribute's values
    DictPropAttr = {}
    for attr in DictFreqAttr:
        DictPropAttr[attr] = {}
        for val in DictFreqAttr[attr]:
            DictPropAttr[attr][val] = DictFreqAttr[attr][val] / N

    # Compute the probability^2 of each attribute's values
    DictPropAttr2 = {}
    for attr in DictFreqAttr:
        DictPropAttr2[attr] = {}
        for val in DictFreqAttr[attr]:
            DictPropAttr2[attr][val] = (
                DictFreqAttr[attr][val]*(
                    DictFreqAttr[attr][val]-1)) / (N*(N-1))

    # Compute Goodall 1
    DictSimAttr = {}
    for attr in DictPropAttr2:
        DictSimAttr[attr] = {}
        for val in DictPropAttr2[attr]:
            DictSimAttr[attr][val] = 1 - sum(
                [DictPropAttr2[attr][x] for x in DictPropAttr2[attr]
                    if DictPropAttr2[attr][x] <= DictPropAttr2[attr][val]])

    # Create Similarity/Distance Matrix
    SimMatrix = np.zeros((N, N))
    for c in combinations(range(N), 2):
        s = 0
        for attr in DictSimAttr:
            if df.iloc[c[0]][attr] == df.iloc[c[1]][attr]:
                s += (1 / df.shape[1]) * DictSimAttr[attr][df.iloc[c[0]][attr]]
        SimMatrix[c[0], c[1]] = SimMatrix[c[1], c[0]] = 1 - s

    # Find min and max values
    min_v = SimMatrix[np.where(SimMatrix > 0)].min()
    max_v = SimMatrix[np.where(SimMatrix > 0)].max()
    min_i = np.where(SimMatrix == min_v)
    max_i = np.where(SimMatrix == max_v)
    # print(min_v, max_v, min_i, max_i)

    # Save Matrix and Plot it
    np.savetxt("data/AD/AD_Goodal1_out_20201105.csv",
               SimMatrix, delimiter=',', fmt='%.4f')
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(SimMatrix, interpolation='nearest',
                         cmap=cm.Spectral_r)
    figure.colorbar(caxes)
    plt.show()

    # Compute Stats for Group
    c = 0
    while c < 32:
        T = np.triu(SimMatrix[c:c+4, c:c+4])
        st = stats.describe(T[T > 0.000000001])
        print("min-max:", st[1])
        print("avg:", st[2])

        v = []
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if j > i:
                    v.append(T[i, j])
        print("hmean:", stats.hmean(v))
        print()
        c += 4
