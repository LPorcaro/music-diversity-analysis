#!/usr/bin/env python
# encoding: utf-8


import numpy as np 
import pandas as pd

known = ["I know them/her/him",
         "Los/la/lo conozco",
         "Li/la/lo conosco"]

unknown = ["I don't know them/her/him",
           "No los/la/lo conozco",
           "Non li/la/lo conosco"]

maybe = ["Maybe I know them/her/him",
         "Quizás los/la/lo conozco",
         "Forse li/la/lo conosco"]

if __name__ == '__main__':

    scores = "../data/Familiarity/InverseGAP0Score.csv"
    infile = "../data/Familiarity/FamiliaritySurvey.csv"

    df_score = pd.read_csv(scores)
    df_res = pd.read_csv(infile)

    u_scores = []
    for index, row in df_res.iterrows():
        u_score = 0 
        for art in df_score.columns.values:
            if row[art] in known:
                score = df_score[art].values[0]
            elif row[art] in maybe:
                score = df_score[art].values[0]/2
            else:
                score = 0
            u_score += score
        u_scores.append(u_score)


    u_scores = [(x-np.min(u_scores))/(np.max(u_scores)-np.min(u_scores)) for x in u_scores]

    for el in u_scores:
        print(el)