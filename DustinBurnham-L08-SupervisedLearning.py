#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:44:57 2019

Dustin Burnham
2/26/2019
Data Science 400
Assignment 8: Predictive Analytics - Intro to Supervised Learning

1. Short narrative on the data preparation for your chosen data set from Milestone 2.
Summary:
    I selected a more numeric data set compared to my dataset from milestone 2.  
    The data set I will work with is the auto dataset which consists of car data.
    To prepare the data I will replace all missing values with the mean of that column
    and drop the column with specific car names as that is random data.  Random
    data will add no value to our model.  Next I will normalize all of the columns
    so they can be compared, and also one-hot-encode the categorical variable 'origin'
    for analysis.

    Classification:
    For the classification, I thought that I would investigate the relationship
    between the horsepower of a car and the number of cylinders.  I initially
    did some unsupervised k-mean clustering to look for clusters.  These clusters
    lined up well with my prediction that they were related.  Knowing that I had
    an expert label in the 'cylinders' column, we could do some supervised clustering.
    I chose random forrest for the reason that it seemed like it would work theoretically
    and the results were fairly accurate.  After training a model with 70-30
    ratio of train-test, the predictions that I generated matched the expert 
    label with an accuracy of around 93%.
"""

##############################################################################
# 2.  Import statements for necessary packages
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

#Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from copy import deepcopy

##############################################################################
# 3. Read in the dataset from a freely and easily available source on the internet.
##############################################################################

# Read in data, add header, drop useless column
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
names = ["mpg", "cylinders", "displacement", "horsepower", 
                "weight", "acceleration", "model_year", "origin", "car_name"]
Auto.columns = names
Auto = Auto.drop("car_name", axis=1)

##############################################################################
# 4. Show data preparation. Normalize some numeric columns, one-hot encode some 
# categorical columns with 3 or more categories, remove or replace missing values, 
# remove or replace some outliers.
##############################################################################

def normalize(X, names): 
    """
    input: 
        X: Pandas Data Frame that will be min-max normalized
        names: column name
        
    Output:
        Y: Normalized data frame
    """
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        name = names[i]
        mX = min(X.loc[:,name])
        Y[:,i] = (X.loc[:,name] - mX) / (max(X.loc[:,name]) - mX)
    
    return Y

# The horsepower column is the only one with missing values, so I will
# impute the missing values ('?') with the numeric mean of the other
# values in that column.
Auto.loc[:, "horsepower"] = pd.to_numeric(Auto.loc[:, "horsepower"], errors='coerce')
HasNan = np.isnan(Auto.loc[:,"horsepower"])
Auto.loc[HasNan, "horsepower"] = np.nanmean(Auto.loc[:,"horsepower"])
Auto.loc[:, "horsepower"] = pd.to_numeric(Auto.loc[:, "horsepower"])

# One-Hot-Encode the origin column into the columns North America, Europe, and Asia.
# Drop the now redundant column.
Auto.loc[:, "North America"] = (Auto.loc[:, "origin"] == 1).astype(int)
Auto.loc[:, "Europe"] = (Auto.loc[:, "origin"] == 2).astype(int)
Auto.loc[:, "Asia"] = (Auto.loc[:, "origin"] == 3).astype(int)
Auto = Auto.drop("origin", axis=1)

# Normalize the numeric columns of the data frame.
normAuto = normalize(Auto.loc[:, names[:-3]], names)

##############################################################################
# 5. Ask a binary-choice question that describes your classification. 
# Write the question as a comment. Specify an appropriate column as your expert 
# label for a classification (include decision comments).  (Is it...? Does it...?)
##############################################################################

# BINARY-CHOICE QUESTION:
# Is the amount of horsepower that a car has related to the amount of cylinders
# that a car has?
# Expert label: Cylinders column (3,4,5,6,8)

##############################################################################
# 6. Apply K-Means on some of your columns, but make sure you do not use the expert 
# label. Add the K-Means cluster labels to your dataset.
##############################################################################

# Function mostly borrowed from lesson 7 lab
def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()
    
# Function mostly borrowed from lesson 7 lab
def KMeansNorm(Points, ClusterCentroidGuesses):
    PointsNorm = Points.copy()
    ClusterCentroids = ClusterCentroidGuesses.copy()
    kmeans = KMeans(n_clusters=3, init=ClusterCentroidGuesses, n_init=1).fit(PointsNorm)
    Labels = kmeans.labels_
    ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
    return Labels, ClusterCentroids

# Generate random cluster centroids.  I will guess 3 clusters for the 
# most abundant types of cylinders (4, 6, 8).
ClusterCentroidGuesses = pd.DataFrame(np.random.rand(3,2))
labels, clusterCentroids = KMeansNorm(pd.DataFrame(normAuto[:,[3,5]]), ClusterCentroidGuesses)

# Plot horsepower vs acceleration and color with the cluster labels to see if
# the clustering is reasonable.  It is.
title = "Clusters of Horsepower vs. Acceleration"
Plot2DKMeans(pd.DataFrame(normAuto[:,[3,5]]), labels, clusterCentroids, title)
Auto.loc[:,"cluster_label"] = labels

# Plot horsepower vs cylinders and color by the cluster labels to see if
# the labels match up well with the cylinders variable.  There is some 
# ambiguity for 4,6 cylinder cars, but the 8 cylinder vehicles fit a
# cluster very well.
plt.scatter(Auto.loc[:,"horsepower"], Auto.loc[:,"cylinders"],c=labels)
plt.xlabel("Horsepower")
plt.ylabel("Cylinder")
plt.title("Horsepower vs Cylinder colored by labels")
plt.show()

##############################################################################
# 7. Split your data set into training and testing sets using the proper function 
# in sklearn (include decision comments).
##############################################################################

# Function borrowed from lab
def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targests
	return X, XX, Y, YY

# 70% training data, 30% test data
r = 0.3
normalized_data = deepcopy(Auto.values)
normalized_data[:,:6] = normAuto # insert normalized columns
normalized_data[:,-1] = normalized_data[:,1]
normalized_data = np.delete(normalized_data, 1, 1) # Remove the expert label
X, XX, Y, YY = split_dataset(normalized_data[:-1], r) # Calculate the test/training features/targets

##############################################################################
# 8. Create a classification model for the expert label based on the training data 
# (include decision comments).
##############################################################################

# Random Forest classifier
# I chose random forest because I knew that I was trying to classify which
# eliminated the regression methods.  Random forest provided accurate results.
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X, Y) # Create model

##############################################################################
# 9. Apply your (trained) classifiers to the test data to predict probabilities.
##############################################################################

print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)

##############################################################################
# 10. Write out to a csv a dataframe of the test data, including actual outcomes, 
# and the probabilities of your classification.
##############################################################################

# Add test, actual targets to data frame and export as csv.
test_data = pd.DataFrame()
test_data.loc[:, "Predictions"] = clf.predict(XX)
test_data.loc[:, "Actual"] = YY

filename = 'DustinBurnham-L08-TestData.csv'
test_data.to_csv(filename)

##############################################################################
# 11. Determine accuracy rate, which is the number of correct predictions divided 
# by the total number of predictions (include brief preliminary analysis commentary).
##############################################################################

def percent(lst1, lst2):
    """
    input: two lists for computing accuracy as a percentage.  Lst1 are testing
    features and lst2 are testing targets.  A percentage float is returned.
    """
    
    return(100 * (sum(clf.predict(lst1) == lst2)) / len(lst1))

accuracy = percent(XX, YY)
print("Accuracy Rate: %", accuracy)

"""
Commentary:
    The accuracy rate of 93% is pretty good for a first attempt, and certainly
    better than the models seen in the lab.  Given other data, we would
    be able to predict the number of cylinders based on the horsepower of a car.
"""
