#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import numpy as np
import math


INPUT = "data/Users/Users_TVFactors_20201028.csv"

if __name__ == '__main__':

    df  = pd.read_csv(INPUT)
    df_new = df.replace(['Strong influence'],3).replace(
                            ['Medium influence'],2).replace(
                                ["Weak influence"],1).replace(
                                    ["No influence"],0)


    df_new.drop(['TV1'], axis=1, inplace=True)
    df_new.drop(['TV4'], axis=1, inplace=True)

    cohen_kappas = []
    for index in list(combinations(df_new.index,2)):
        k = cohen_kappa_score(df_new.loc[index[0]], df_new.loc[index[1]])
        if math.isnan(k):
            k=1
        cohen_kappas.append(k)


    print(np.mean(cohen_kappas))