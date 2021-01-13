#!/usr/bin/env python
# encoding: utf-8

from scipy.stats.stats import pearsonr
import pandas as pd


INPUT_FILE = "../data/Users/Users_correlation.csv"


G1 = G2 = ["FormalTraining", 
           "PlayiningTime", 
           "TasteVariety", 
           "ElectronicMusic", 
           "ElectronicTasteVariety", 
           "GAPScore"]


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
                    print((g1,g2,ro, p))
                    done.add((g1,g2))
                    done.add((g2,g1))



