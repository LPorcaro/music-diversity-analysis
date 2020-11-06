#!/usr/bin/env python
# encoding: utf-8

import csv
import numpy as np 
import gower
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import cohen_kappa_score

from kripp_juan import alpha as k_alpha 

INPUT_FEED = "data/TV/TrackVarietyAnswersB.csv"
# INPUT_FEED = "data/AD/ArtistDiversityAnswersB.csv"
# INPUT_FEED = "data/MIX/MixedAnswersB.csv"
INPUT_USERS = "data/Users/Users_20201029.csv"
VALUES_DOMAIN = ["List A", "List B", "I don't know"]    

MIX_FLAG = False

MVALUES_TV = [2,1,1,2]
MVALUES_AD = [2,2,1,2]
MVALUES_MIX = [[2,1,2,1],
               [2,2,2,1],
               [2,2,2,2],
               [2,2,1,2],
               [2,2,1,2]]

def SilhouetteAnalysis(X):
    """
    """
    range_n_clusters = np.arange(2,20)
    silhouette_avgs = []
    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMedoids(n_clusters=n_clusters,
                             metric="precomputed",
                             random_state=0)
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        silhouette_avgs.append(silhouette_avg)


    plt.plot(range_n_clusters, silhouette_avgs)
    locs, labels=plt.xticks()
    new_xticks=[str(x) for x in range_n_clusters]
    plt.xticks(range_n_clusters, new_xticks, rotation=45, horizontalalignment='right')
    plt.show()

    cluster_number = silhouette_avgs.index(max(silhouette_avgs)) + 2
    print("\nOptimal number of clusters = {}".format(cluster_number))
    return cluster_number

def compute_alpha(participant_ids):
    """
    """
    k_a, o = k_alpha(df_feedback.iloc[participant_ids],
                     value_domain=VALUES_DOMAIN, 
                     level_of_measurement='nominal')
    print(k_a)#, len(participant_ids))
    return k_a

def compute_cohen(participant_ids):
    """
    """
    print(len(participant_ids))
    # df_feedbacknew = df_feedback.replace(['List B'],1).replace(['List A'],2).replace(["I don't know"],0)
    df_feedbacknew = df_feedback

    k_tot = []
    for co in combinations(participant_ids, 2):
        k = cohen_kappa_score(df_feedbacknew.iloc[co[0]], df_feedbacknew.iloc[co[1]])
        if math.isnan(k):
            k=1
        k_tot.append(k)

    c_k = np.mean(k_tot)

    # print(c_k)
    return c_k

def compute_cohen_metric(participant_ids, metric_values, mix):
    """
    """
    df_feedbacknew = df_feedback.replace(['List A'],1).replace(['List B'],2).replace(["I don't know"],0)

    k_tot = []


    for c in participant_ids:
        if mix:
            metric_values = MVALUES_MIX[df_users.iloc[c]['MIX2']-1]

        metric_values = MVALUES_TV[2:]
        k = cohen_kappa_score(df_feedbacknew.iloc[c], metric_values)
        if math.isnan(k):
            k=1
        k_tot.append(k)

    c_k = np.mean(k_tot)

    print(c_k)
    return c_k

def create_users_group(df_users):
    """
    """
    # 1 - Participants Musical Education = Yes
    users_ME_True = list(df_users.loc[
                            df_users["MusicalEducation"]=="Yes"][
                                "MusicalEducation"].to_dict().keys())
    users_groups.append(users_ME_True)

    # 2 - Participants Musical Education = No
    users_ME_False = list(df_users.loc[
                            df_users["MusicalEducation"]=="No"][
                                "MusicalEducation"].to_dict().keys())
    users_groups.append(users_ME_False)
    

    # 3 - Participants Playing Time >4
    users_PT_1 = list(df_users.loc[
                            df_users["PlayiningTime"]>3][
                                "PlayiningTime"].to_dict().keys())
    users_groups.append(users_PT_1)

    # 4 - Participants Playing Time <4
    users_PT_2 = list(df_users.loc[
                            df_users["PlayiningTime"]<=3][
                                "PlayiningTime"].to_dict().keys())
    users_groups.append(users_PT_2)

    # 5 - Participants Taste Variety == 5
    users_TV_1 = list(df_users.loc[
                            df_users["TasteVariety"]>4][
                                "TasteVariety"].to_dict().keys())
    users_groups.append(users_TV_1)

    # 6- Participants Taste Variety < 5
    users_TV_1 = list(df_users.loc[
                            df_users["TasteVariety"]<=4][
                                "TasteVariety"].to_dict().keys())
    users_groups.append(users_TV_1)

    # 7 - Participants ElectronicMusic == 5
    users_TV_1 = list(df_users.loc[
                            df_users["ElectronicMusic"]==5][
                                "ElectronicMusic"].to_dict().keys())
    users_groups.append(users_TV_1)

    # 8- Participants ElectronicMusic < 5
    users_TV_1 = list(df_users.loc[
                            df_users["ElectronicMusic"]<5][
                                "ElectronicMusic"].to_dict().keys())
    users_groups.append(users_TV_1)

    # 9 - Participants ElectronicTasteVariety >4
    users_PT_1 = list(df_users.loc[
                            df_users["ElectronicTasteVariety"]>4][
                                "ElectronicTasteVariety"].to_dict().keys())
    users_groups.append(users_PT_1)

    # 10 - Participants ElectronicTasteVariety <4
    users_PT_2 = list(df_users.loc[
                            df_users["ElectronicTasteVariety"]<=4][
                                "ElectronicTasteVariety"].to_dict().keys())
    users_groups.append(users_PT_2)


    mean_main = df_users["ScoreMainstream"].mean()

    # 11 - Participants ScoreMainstream >= median_sur
    users_SM_1 = list(df_users.loc[
                            df_users["ScoreMainstream"]>mean_main][
                                "ScoreMainstream"].to_dict().keys())
    users_groups.append(users_SM_1)

    # 12 - Participants ElectronicTasteVariety < median_sur
    users_SM_2 = list(df_users.loc[
                            df_users["ScoreMainstream"]<=mean_main][
                                "ScoreMainstream"].to_dict().keys())
    users_groups.append(users_SM_2)

    mean_sur = df_users["ScoreSurvey"].mean()

    # 13 - Participants ScoreSurvey >= median_sur
    users_SS_1 = list(df_users.loc[
                            df_users["ScoreSurvey"]>mean_sur][
                                "ScoreSurvey"].to_dict().keys())
    users_groups.append(users_SS_1)

    # 14 - Participants ScoreSurvey < median_sur
    users_SS_2 = list(df_users.loc[
                            df_users["ScoreSurvey"]<=mean_sur][
                                "ScoreSurvey"].to_dict().keys())
    users_groups.append(users_SS_2)    


    # Demographical Clusters
    WEIRD1 = df[(df["Gender"]=="Male") &
                   ((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
                   (df["Instruction"] != "High school degree or equivalent") & 
                   (df["SkinType"].isin(["1","2"]))]

    users_groups.append(WEIRD1.index.to_list())
    

    WEIRD2 = df[((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
                   (df["Instruction"] != "High school degree or equivalent") & 
                   (df["SkinType"].isin(["1","2"]))]

    users_groups.append(WEIRD2.index.to_list())
    

    WEIRD3 = df[((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
                   (df["Instruction"] != "High school degree or equivalent")]

    users_groups.append(WEIRD3.index.to_list())
    

    YOUNG = df[df["Age"].isin(["18-24 years old","25-34 years old"])]
    OLD = df[~df.isin(YOUNG)].dropna()
    users_groups.append(YOUNG.index.to_list())
    users_groups.append(OLD.index.to_list())

    return users_groups
    

if __name__ == '__main__':

    # Import Data 
    df = pd.read_csv("data/Users/Users_dem_20201014.csv")
    df_users = pd.read_csv(INPUT_USERS)
    df_feedback = pd.read_csv(INPUT_FEED)

    # Compute Distance Matrix
    DistMatrix = gower.gower_matrix(df_users)

    users_groups = []
    # Create Pre-defined users group
    # users_groups = create_users_group(df_users)

    # SILHOUETTE ANALYSIS
    # n_clusters = SilhouetteAnalysis(DistMatrix)
    n_clusters = 3

    # GOWER DISTANCE
    # clusterer = KMedoids(n_clusters=n_clusters,
    #                      metric="precomputed",
    #                      random_state=0)
    # clusterer.fit_transform(DistMatrix)
    # # Plot TSNE 
    # model = TSNE(n_components=2, 
    #             metric="precomputed",
    #             random_state=0)
    # Y = model.fit_transform(DistMatrix)
    # plt.scatter(Y[:, 0], Y[:, 1], c=clusterer.labels_, cmap=plt.cm.Spectral)
    # plt.show()


    # EUCLIDEAN DISTANCE
    clusterer = KMedoids(n_clusters=n_clusters,
                         metric="euclidean",
                         random_state=0)
    clusterer.fit_transform(df_users)


    # Plot TSNE 
    model = TSNE(n_components=2, 
                metric="euclidean",
                random_state=0)
    Y = model.fit_transform(df_users)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=clusterer.labels_, cmap=plt.cm.rainbow, label=clusterer.labels_, s=50)
    handles, labels = scatter.legend_elements()
    legend1 = ax.legend(handles,["Medium Soph", "Low Soph", "High Soph"],
                        loc="lower left", title="Classes",  prop={'size': 20})
    ax.add_artist(legend1)
    plt.show()


    # for n_c in np.arange(0, n_clusters):
    #     print(n_c)
    #     print(df_users.iloc[np.where(clusterer.labels_==n_c)].describe(include='all'))
    #     df_users.iloc[np.where(clusterer.labels_==n_c)].describe(include='all').to_csv("data/Users/user_cluster_20201029_{}.csv".format(n_c))
    #     group = np.where(clusterer.labels_==n_c)[0]
    #     users_groups.append(group)
    

    
    # INTER-RATER AGREEMENT
    # Group Agreement
    # print("Group Agreement")
    
    # for group in users_groups:
    #     compute_cohen(group)
    
    # # Overall Agreement
    # # print("Overall Agreement")
    # compute_cohen(list(df_users['MusicalEducation'].to_dict().keys()))

    # # Agreement among combinations of 2 groups
    # print()
    # for group in combinations(users_groups,2):
    #     combo = list(set(group[0]) & set(group[1]))
    #     compute_alpha(combo)


    # # METRIC-RATER AGREEMENT
    # metric_values = []
    # for group in users_groups:
    #     compute_cohen_metric(group, metric_values, MIX_FLAG)

    # compute_cohen_metric(df_users.index.to_list(), metric_values, MIX_FLAG)
    

