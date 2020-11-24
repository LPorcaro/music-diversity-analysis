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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
from numpy.linalg import norm

from kripp_juan import alpha as k_alpha 

# INPUT_FEED = "data/TV/TrackVarietyAnswersB.csv"
INPUT_FEED = "data/AD/ArtistDiversityAnswersB.csv"
# INPUT_FEED = "data/MIX/MixedAnswersB.csv"

INPUT_USERS = "data/Users/Users_soph_fam_20201029.csv"
# INPUT_USERS = "data/Users/Users_Factors_20201117.csv"

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
                             metric="euclidean",
                             random_state=rng)
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
    k_tot = []
    for co in combinations(participant_ids, 2):
        k = cohen_kappa_score(df_feedback.iloc[co[0]], df_feedback.iloc[co[1]])
        if math.isnan(k):
            k=1
        k_tot.append(k)

    c_k = np.mean(k_tot)

    print(c_k)
    # print(len(Participantsicipant_ids), c_k)
    return c_k

def compute_cohen_metric(participant_ids, mix):
    """
    """
    df_feedbacknew = df_feedback.replace(
                        ['List A'],1).replace(
                            ['List B'],2).replace(
                                ["I don't know"],0)

    k_tot = []

    for c in participant_ids:
        if mix:
            metric_values = MVALUES_MIX[df_users.iloc[c]['MIX2']-1]
        else:
            metric_values = MVALUES_AD[2:]
            
        k = cohen_kappa_score(df_feedbacknew.iloc[c], metric_values)
        if math.isnan(k):
            k=1
        k_tot.append(k)

    c_k = np.mean(k_tot)

    print(c_k)
    return c_k

def create_users_groups(df):
    """
    """
    FormalActivePlayers = df[(df["MusicalEducation"]=="Yes") & (df["PlayiningTime"]>=4)]
    users_groups.append(FormalActivePlayers.index.to_list())

    FormalUnactivePlayers = df[(df["MusicalEducation"]=="Yes") & (df["PlayiningTime"]<4)]
    users_groups.append(FormalUnactivePlayers.index.to_list())

    InformalActivePlayers = df[(df["MusicalEducation"]=="No") & (df["PlayiningTime"]>=4)]
    users_groups.append(InformalActivePlayers.index.to_list())

    InformalUnactivePlayers = df[(df["MusicalEducation"]=="No") & (df["PlayiningTime"]<=4)]
    users_groups.append(InformalUnactivePlayers.index.to_list())


    # Demographical Clusters
    WEIRD1 = df[(df["Gender"]=="Male") &
                   ((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
                   ((df["Instruction"] != "High school degree or equivalent") | (df["Instruction"] != "Less than a high school diploma")) & 
                   (df["SkinType"].isin(["1","2"]))]

    users_groups.append(WEIRD1.index.to_list())
    

    WEIRD2 = df[((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
                   ((df["Instruction"] != "High school degree or equivalent") | (df["Instruction"] != "Less than a high school diploma")) & 
                   (df["SkinType"].isin(["1","2"]))]

    users_groups.append(WEIRD2.index.to_list())
    

    WEIRD3 = df[((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
                   ((df["Instruction"] != "High school degree or equivalent") | (df["Instruction"] != "Less than a high school diploma"))]

    users_groups.append(WEIRD3.index.to_list())
    

    YOUNG = df[df["Age"].isin(["18-24 years old","25-34 years old"])]
    users_groups.append(YOUNG.index.to_list())

    OLD = df[~df.isin(YOUNG)].dropna()
    users_groups.append(OLD.index.to_list())

    return users_groups
    
def cluster_points(X):
    """
    """
    # n_clusters = SilhouetteAnalysis(df_users) # SILHOUETTE ANALYSIS

    clusterer = KMedoids(n_clusters=n_clusters,
                         metric=metric,
                         random_state=rng)
    clusterer.fit_transform(X)

    c1, c2, c3 = clusterer.medoid_indices_

    # Remove outliers
    c1_dist = [norm(X.iloc[c1]-X.iloc[x]) for x in np.setdiff1d(np.where(clusterer.labels_ == 0)[0], [c1])] 
    c2_dist = [norm(X.iloc[c2]-X.iloc[x]) for x in np.setdiff1d(np.where(clusterer.labels_ == 1)[0], [c2])] 
    c3_dist = [norm(X.iloc[c3]-X.iloc[x]) for x in np.setdiff1d(np.where(clusterer.labels_ == 2)[0], [c3])] 

    c1_avg = np.mean(c1_dist)
    c2_avg = np.mean(c2_dist)
    c3_avg = np.mean(c3_dist)
    c1_std = np.std(c1_dist)
    c2_std = np.std(c2_dist)
    c3_std = np.std(c3_dist)

    outliers = []
    for c, el in enumerate(clusterer.labels_):
        if el == 0:
            if norm(X.iloc[c1]-X.iloc[c]) > c1_avg + c1_std:
                outliers.append(c)
        elif el == 1:
            if norm(X.iloc[c2]-X.iloc[c]) > c2_avg + c2_std:
                outliers.append(c)
        elif el == 2:
            if norm(X.iloc[c3]-X.iloc[c]) > c3_avg + c3_std:
                outliers.append(c)

    X_new = X.drop(outliers)

    clusterer_new = KMedoids(n_clusters=n_clusters,
                             metric=metric,
                             random_state=rng)
    clusterer_new.fit_transform(X_new)

    # plot_clusters(X, X_new, clusterer, clusterer_new)

    return clusterer_new

def plot_clusters(X, X_new, clusterer, clusterer_new):
    """
    """
    fig, axs = plt.subplots(2, 2)
    ### Plot TSNE ###
    tsne = TSNE(n_components=2, 
                metric=metric,
                random_state=rng)

    Y = tsne.fit_transform(X)
    scatter = axs[0,0].scatter(Y[:, 0], Y[:, 1], c=clusterer.labels_, cmap=plt.cm.rainbow, label=clusterer.labels_, s=50)
    # handles, labels = scatter.legend_elements()
    # legend1 = ax1.legend(handles,["Medium Soph", "Low Soph", "High Soph"], loc="lower left", title="Classes",  prop={'size': 20})
    # ax1.add_artist(legend1)

    ### Plot PCA ###
    pca = PCA(n_components=2)
    Y2 = pca.fit_transform(X)
    scatter2 = axs[0,1].scatter(Y2[:, 0], Y2[:, 1], c=clusterer.labels_, cmap=plt.cm.rainbow, label=clusterer.labels_, s=50)
    # legend1 = ax2.legend(handles,["Medium Soph", "Low Soph", "High Soph"], loc="lower left", title="Classes",  prop={'size': 20})
    # ax2.add_artist(legend1)


    Y3 = tsne.fit_transform(X_new)
    scatter3 = axs[1,0].scatter(Y3[:, 0], Y3[:, 1], c=clusterer_new.labels_, cmap=plt.cm.rainbow, label=clusterer_new.labels_, s=50)
    # handles, labels = scatter.legend_elements()
    # legend1 = ax1.legend(handles,["Medium Soph", "Low Soph", "High Soph"], loc="lower left", title="Classes",  prop={'size': 20})
    # ax1.add_artist(legend1)


    ### Plot PCA ###
    pca = PCA(n_components=2)
    Y4 = pca.fit_transform(X_new)
    scatter4 = axs[1,1].scatter(Y4[:, 0], Y4[:, 1], c=clusterer_new.labels_, cmap=plt.cm.rainbow, label=clusterer_new.labels_, s=50)
    # legend1 = ax2.legend(handles,["Medium Soph", "Low Soph", "High Soph"], loc="lower left", title="Classes",  prop={'size': 20})
    # ax2.add_artist(legend1)
    plt.show()


if __name__ == '__main__':

    # Import Data 
    df_users = pd.read_csv("data/Users/Users_dem_20201118.csv")
    df_users_clust = pd.read_csv(INPUT_USERS)
    df_feedback = pd.read_csv(INPUT_FEED)

    users_groups = []
    # Create Pre-defined users group
    users_groups = create_users_groups(df_users)

    rng = np.random.RandomState(42)
    metric = "euclidean"
    n_clusters = 3
    clusterer = cluster_points(df_users_clust)

    for n_c in np.arange(0, n_clusters):
        # print(n_c)
        # print(df_users_clust.iloc[np.where(clusterer.labels_==n_c)].describe(include='all'))
        # df_users_clust.iloc[np.where(clusterer.labels_==n_c)].describe(include='all').to_csv("data/Users/user_cluster_20201029_{}.csv".format(n_c))
        group = np.where(clusterer.labels_ == n_c)[0]
        users_groups.append(group)


    
    # ### INTER-RATER AGREEMENT ###
    # for group in users_groups:
    #     compute_cohen(group)
    # compute_cohen(df_users.index.to_list())


    ### METRIC-RATER AGREEMENT ###
    for group in users_groups:
        compute_cohen_metric(group, MIX_FLAG)
    compute_cohen_metric(df_users.index.to_list(), MIX_FLAG)
    

