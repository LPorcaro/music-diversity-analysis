#!/usr/bin/env python
# encoding: utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math
import random

from itertools import combinations
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from numpy.linalg import norm


# INPUT_FEED = "data/TV/TrackVarietyAnswersB.csv"
# INPUT_FEED = "data/AD/ArtistDiversityAnswersB.csv"
INPUT_FEED = "data/MIX/MixedAnswersB.csv"

INPUT_USERS = "data/Users/Users_soph_fam_20201029.csv"
# INPUT_USERS = "data/Users/Users_Factors_20201117.csv"

VALUES_DOMAIN = ["List A", "List B", "I don't know"]    


MVALUES_TV = ['List B','List A','List A','List B']
MVALUES_AD = ['List B','List B','List A','List B']
MVALUES_MIX = ['List B','List B','List B','List A']


def compute_ci(kappa, group):
    """
    """
    resample = 100

    kappas = []
    for i in range(resample):
        ids_resampled = np.array(random.choices(group, k=len(group)))
        kappas.append(compute_cohen(ids_resampled))

    margin = 1.96*np.std(kappas)/np.sqrt(len(group))
    print("{:.2f} \pm {:.2f}".format(kappa, margin))

def compute_ci_metric(kappa, group):
    """
    """
    resample = 100

    kappas = []
    for i in range(resample):
        ids_resampled = np.array(random.choices(group, k=len(group)))
        kappas.append(compute_cohen_metric(ids_resampled))

    margin = 1.96*np.std(kappas)/np.sqrt(len(group))
    print("{:.2f} \pm {:.2f}".format(kappa, margin))

def compute_cohen(participant_ids):
    """
    """
    k_tot = []
    for co in combinations(participant_ids, 2):
        k = cohen_kappa_score(df_feedback.iloc[co[0]], 
                              df_feedback.iloc[co[1]])
        if math.isnan(k):
            k=1
        k_tot.append(k)

    c_k = np.mean(k_tot)

    return c_k

def compute_cohen_metric(participant_ids):
    """
    """
    k_tot = []

    for c in participant_ids:            
        k = cohen_kappa_score(df_feedback.iloc[c], MVALUES_MIX[2:])
        if math.isnan(k):
            k=1
        k_tot.append(k)

    c_k = np.mean(k_tot)

    # print(c_k)
    return c_k

def create_users_groups(df):
    """
    """
    # G1 = df[df["MIXChoice"]=="Now"]
    # users_groups.append(G1.index.to_list())    

    # G2 = df[df["MIXChoice"]=="Previous"]
    # users_groups.append(G2.index.to_list())    

    # Demographical Clusters
    # WEIRD1 = df[(df["Gender"]=="Male") &
    #                ((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
    #                ((df["Instruction"] != "High school degree or equivalent") | (df["Instruction"] != "Less than a high school diploma")) & 
    #                (df["SkinType"].isin(["1","2"]))]

    # users_groups.append(WEIRD1.index.to_list())
    

    # WEIRD2 = df[((df["Origin"]=="Europe") | (df["Origin"]=="North America") | (df["Origin"]=="Oceania")) &
    #                ((df["Instruction"] != "High school degree or equivalent") | (df["Instruction"] != "Less than a high school diploma")) & 
    #                (df["SkinType"].isin(["1","2"]))]

    # users_groups.append(WEIRD2.index.to_list())
    

    WEIRD3 = df[((df["Origin"]=="Europe") | (df["Origin"]=="North America")) &
                   ((df["Instruction"] != "High school degree or equivalent") | (df["Instruction"] != "Less than a high school diploma"))]
    print(len(WEIRD3.index.to_list()))
    users_groups.append(WEIRD3.index.to_list())

    NOT_WEIRD = df[~df.isin(WEIRD3)].dropna()
    users_groups.append(NOT_WEIRD.index.to_list())    
    print(len(NOT_WEIRD.index.to_list()))

    YOUNG = df[df["Age"].isin(["18-24 years old","25-34 years old"])]
    users_groups.append(YOUNG.index.to_list())
    print(len(YOUNG.index.to_list()))

    OLD = df[~df.isin(YOUNG)].dropna()
    users_groups.append(OLD.index.to_list())
    print(len(OLD.index.to_list()))

    return users_groups
    
def cluster_points(X):
    """
    """

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

    rng = np.random.RandomState(42)
    metric = "euclidean"
    n_clusters = 3
    clusterer = cluster_points(df_users_clust)

    print("\n### Create groups and print size")
    for n_c in np.arange(0, n_clusters):
        group = np.where(clusterer.labels_ == n_c)[0]
        users_groups.append(group)
        print(len(group))


    # Create Pre-defined users group
    users_groups = create_users_groups(df_users)

    
    ### INTER-RATER AGREEMENT ###
    print("\n### Compute Inter-Rater Agreement")
    for group in users_groups:
        c_k = compute_cohen(group)
        compute_ci(c_k, group)
    c_k = compute_cohen(df_users.index.to_list())
    compute_ci(c_k, df_users.index.to_list())


    ### METRIC-RATER AGREEMENT ###
    print("\n### Compute Metric Rater Agreement")
    for group in users_groups:
        c_k = compute_cohen_metric(group)
        compute_ci_metric(c_k, group)
    c_k = compute_cohen_metric(df_users.index.to_list())
    compute_ci_metric(c_k, df_users.index.to_list())