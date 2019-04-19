#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:43:41 2019

Dustin Burnham
2/19/19
Data Science 400
Assignment 7: Unsupervised Learning with K-Means

Summary:
    The cluster labels that I saved was specifically between the variables
    of age vs hours-per-week and hours-per-week vs education.  I believe
    these numeric variables are related, and thus would be a good candidate 
    for clustering using k-means.  The labels have been saved as attributes
    in the original data frame.  I clustered with the one-hot-encoded attributes
    for race, but I found these less helpful as there was no great clustering
    that I could see.  When I selected for specific races and investigated
    age vs hours per week, the clustering was slightly different.  By plotting
    other variables in the future, we can color the points using these labels
    to see how the data groups might affect other attributes.
"""

"""
1. Short narrative on the data preparation for your chosen data set for Milestone 3, 
which in most cases should be the same as Milestone 2.
    -number of observations and attributes:
        18 Attributes, 30162 observations after cleaning.
    -datatype, distribution, and a comment on each attribute:
        I will use all numerica data or categorical data that has been one-hot encoded
        Attributes:
            age->numeric, unimodal, peak around 40
            education-num->numeric, bimodal, peaks around 10, 14
            hours-per-week–>numeric, unimodal, strong peak around 35-40
            white,black,asian-pac-islander,american-indian-eskimo, other-> one-hot-encoded race attribute, binary
    -Source citation for your data set
        https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    -Ask at least 1 yes-no or binary-choice question (Does it...? Is it...?)
        Question 1: Is the level of education related to the hours per week?
        Answer: Yes.  The clustering suggests that those with low education
        tend to work around exactly 40 hours per week, while the more highly educated
        individuals tend to work a larger variety of hours.
    -Ask at least 1 non-binary question (What is...? How many...? When does...?)
        Question 2: What is the relationship between race and hours worked vs age?
        Answer: When looking at the clustering for african americans vs white individuals
        the distributions are slightly different.  We can tell becuase the labeling
        in the clusters is slightly different.  The labeling is comparable becuase
        clustering occurs with the same initial guesses.
"""

# Import packages
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load and clean data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'income']

census = pd.read_csv(url, header=None)
census = census.replace(" ?", float("nan"))
census = census.dropna(axis=0)
census.columns = names

census.loc[:, "White"] = (census.loc[:, "race"] == " White").astype(int)
census.loc[:, "Black"] = (census.loc[:, "race"] == " Black").astype(int)
census.loc[:, "Asian-Pac-Islander"] = (census.loc[:, "race"] == " Asian-Pac-Islander").astype(int)
census.loc[:, "Amer-Indian-Eskimo"] = (census.loc[:, "race"] == " Amer-Indian-Eskimo").astype(int)
census.loc[:, "Other"] = (census.loc[:, "race"] == " Other").astype(int)

# Functions for using k-means clustering
# Plotting function
def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    name1,name2 = list(Points)
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,name1], Points.loc[LabelFlag,name2],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()

# Z normalization function
def normalize(X):
    Y = (X - np.mean(X))/np.std(X)
    return Y

# Z denormalization
def denormalize(X, mean, std):
    Y = (X*std + mean)
    return(Y)

# Normalize points, centroid guesses and perform k-means clustering
# using skLearn KMeans.  Returns the centroids, and labels.  
# Most of this function was borrowed from the lab.
def KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2):
    PointsNorm = Points.copy()
    ClusterCentroids = ClusterCentroidGuesses.copy()
    k = len(ClusterCentroids)
    name1,name2 = list(Points)
    dim1 = Points.loc[:,name1]
    dim2 = Points.loc[:,name2]

    if NormD1:
        # Determine mean of 1st dimension
        mean1 = np.mean(dim1)
        
        # Determine standard deviation of 1st dimension
        sd1 = np.std(dim1)        
        
        # Normalize 1st dimension of Points
        PointsNorm.loc[:,name1] = normalize(dim1)
        
        # Normalize 1st dimension of ClusterCentroids
        ClusterCentroids.loc[:,0] = normalize(ClusterCentroids.loc[:,0])
        
    if NormD2:
        # Determine mean of 1st dimension
        mean2 = np.mean(dim2)
        
        # Determine standard deviation of 1st dimension
        sd2 = np.std(dim2)        
        
        # Normalize 1st dimension of Points
        PointsNorm.loc[:,name2] = normalize(dim2)
        
        # Normalize 1st dimension of ClusterCentroids
        ClusterCentroids.loc[:,1] = normalize(ClusterCentroids.loc[:,1])
        
    # Do actual clustering
    kmeans = KMeans(k, init=ClusterCentroidGuesses, n_init=1).fit(PointsNorm)
    Labels = kmeans.labels_
    ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
    if NormD1:
        # Denormalize 1st dimension
        ClusterCentroids.loc[:,0] = denormalize(ClusterCentroids.loc[:,0], mean1, sd1)
    if NormD2:
        # Denormalize 2nd dimension
        ClusterCentroids.loc[:,1] = denormalize(ClusterCentroids.loc[:,1], mean2, sd2)
    return Labels, ClusterCentroids

# Perform a K-Means with sklearn using some or all of your attributes.
# Cluster for age and hours per week (both numeric).  Only use 1000 observations for 
# plottin purposes.
    
centGuess = pd.DataFrame()
centGuess.loc[:,0] = [20, 40, 60]
centGuess.loc[:,1] = [10, 30, 50]
ageEdu = census.loc[:,['age', 'hours-per-week']]

NormD1=True
NormD2=True
Labels1, ClusterCentroids = KMeansNorm(ageEdu, centGuess, NormD1, NormD2)
Title = 'Age vs education'
Plot2DKMeans(ageEdu.head(1000), Labels1[0:999], ClusterCentroids, Title)

# Cluster for education number and hours per week (both numeric).  
# Only use 1000 observations for plottin purposes.
centGuess = pd.DataFrame()
centGuess.loc[:,0] = [20, 40, 60]
centGuess.loc[:,1] = [7, 10, 12]
workEdu = census.loc[:,['hours-per-week', 'education-num']]

NormD1=True
NormD2=True
Labels2, ClusterCentroids = KMeansNorm(workEdu, centGuess, NormD1, NormD2)
Title = 'Hours per week vs education'
Plot2DKMeans(workEdu.head(1000), Labels2[0:999], ClusterCentroids, Title)

# Include at least one categorical column and one numeric attribute.
# Begin by investigating age and hours per week for both white and black
# people in the study.  Clustering works, but not very clear.  Clustering using
# The one hot encoded data is difficult because the data is binary.  No obvious
# clustering.
centGuess = pd.DataFrame()
centGuess.loc[:,0] = [20, 40, 60]
centGuess.loc[:,1] = [0, 1, 1]
whiteFlag = census.loc[:,'White'] == 1
blackFlag = census.loc[:,'Black'] == 1

ageHoursWhite = census.loc[whiteFlag,['age', 'hours-per-week']]
ageHoursBlack = census.loc[blackFlag,['age', 'hours-per-week']]
ageWhite = census.loc[:, ['age', 'White']]

# Whites individuals
NormD1=True
NormD2=True
Labels, ClusterCentroids = KMeansNorm(ageHoursWhite, centGuess, NormD1, NormD2)
Title = 'Hours per week vs Age White'
Plot2DKMeans(ageHoursWhite.head(1000), Labels[0:999], ClusterCentroids, Title)

# African American individuals
NormD1=True
NormD2=True
Labels, ClusterCentroids = KMeansNorm(ageHoursBlack, centGuess, NormD1, NormD2)
Title = 'Hours per week vs Age African American'
Plot2DKMeans(ageHoursBlack.head(1000), Labels[0:999], ClusterCentroids, Title)

# age vs. being white.  Not very helpful.
centGuess = pd.DataFrame()
centGuess.loc[:,0] = [20, 40, 60, 27, 44]
centGuess.loc[:,1] = [0, 1, 1, 0, 1]

NormD1=True
NormD2=True
Labels, ClusterCentroids = KMeansNorm(ageWhite, centGuess, NormD1, NormD2)
Title = 'Age vs Race'
Plot2DKMeans(ageWhite, Labels, ClusterCentroids, Title)

# Add the cluster label to the dataset.
# Add the labels to the database as new attributes.  Can be plotted against
# other variables to look how this clustering might affect other attributes.
census.loc[:,'age-education-clusters'] = Labels1
census.loc[:, 'hours-per-week-education-clusters'] = Labels2