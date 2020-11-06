#!/usr/bin/env python
# encoding: utf-8


import csv 
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

    scores = "data/Fam/FamiliarityScore_20201020.csv"
    infile = "data/Fam/FamiliaritySurvey_20201028.csv"
    outfile = "data/Fam/FamiliaritySurvey_20201028_out.csv"

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



    # with open(infile, 'r') as inf, open(outfile, 'w+') as outf:
    #     _reader = csv.reader(inf)
    #     _writer = csv.writer(outf)
    #     for row in _reader:
    #         gen_pos = len([x for x in row[:15] if x in known])
    #         gen_neu = len([x for x in row[:15] if x in maybe])
    #         gen_neg = len([x for x in row[:15] if x in unknown])
    #         sur_pos = len([x for x in row[15:] if x in known])
    #         sur_neu = len([x for x in row[15:] if x in maybe])
    #         sur_neg = len([x for x in row[15:] if x in unknown])

    #         gen_fam = (gen_pos-gen_neg)/15
    #         sur_fam = (sur_pos-sur_neg)/15


    #         gen_fam = (gen_pos*1+gen_neu*0.33)/15
    #         sur_fam = (sur_pos*1+sur_neu*0.33)/15

    #         row = [str(x) for x in [gen_pos, gen_neu, gen_neg, sur_pos, sur_neu, sur_neg, gen_fam, sur_fam, (gen_fam+sur_fam)/2]]

    #         # print(','.join(row))
    #         _writer.writerow(row)

