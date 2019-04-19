#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dustin Burnham
3/12/19
Data Science 400
Milestone 3: Data Model and Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from copy import deepcopy

##############################################################################
# Preparation of Data Set
##############################################################################

# 1. Source Citation for your data set #
"""
This data set is car data used in the 1983 American Statistical Association 
Exposition.  Data can be found at:
https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
"""

# 2. Data read from an easily and freely accessible source
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
names = ["mpg", "cylinders", "displacement", "horsepower", 
                "weight", "acceleration", "model_year", "origin", "car_name"]
Auto.columns = names
Auto = Auto.drop("car_name", axis=1)

# 3. Number of observations and attributes
observations, attributes = Auto.shape
print("Observations:", observations)
print("Attributes:", attributes)

# 4. Data types
print("Data Types:\n", Auto.dtypes)

# 5. Distribution of numerical variables
plt.hist(Auto.loc[:, "mpg"])
plt.title("mpg histagram")
plt.show()
plt.hist(Auto.loc[:, "displacement"])
plt.title("displacement histagram")
plt.show()
plt.hist(Auto.loc[:, "weight"])
plt.title("weight histagram")
plt.show()
plt.hist(Auto.loc[:, "acceleration"])
plt.title("acceleration histagram")
plt.show()

"""
Distribution of Numerical Variables:
    mpg: Non-symmetric unimodal distribution centered around 17 mpg.
    displacement: Tri-modal?  Peaks at 100, 250, 320.  Very non-symmetric.
    weight: Non-symmetric unimodal distribtion centered around 2000 pounds.
    acceleration: Symmetric unimodal distribution centered around 15.
"""

# 6.  Distribution of categorical variables
plt.hist(Auto.loc[:,"cylinders"])
plt.title("cylinders distribution")
plt.show()
plt.hist(Auto.loc[:, "model_year"])
plt.title("model year distribution")
plt.show()
plt.hist(Auto.loc[:, "origin"])
plt.title("origin distribution")
plt.show()

"""
Distribution of Categorical Variables:
    cylinders: The distribution shows that cylinders come in values of 3, 4, 5
    6, and 8 with the majority coming in 4, 6, 8.
    model_year: The distribution is nearly uniform between 1970 and 1982 with
    some spikes in 1970, 1976, and 1982.
    origin: There are three possible origins; 1, 2 and 3.  They decode to 
    North America, Europe, and Asia respectively.  North America has significantly
    more compared to Europe and Asia, which are about equal.
"""

# 7. A comment on each attribute
"""
mpg: Continuous numeric values representing miles per gallon.  No missing values or outliers.
cylinders: Discrete numeric values representing number of cylinders in the engine.  
No missing values or outliers.
displacement: Continuous numeric values.  No missing data or outliers.
horsepower: Continuous numeric values.  Contains missing values denoted by '?'.
weight: Continuous numerical values.  No missing data or outliers.
acceleration: Continuous numerical values.  No missing data or outliers.
model_year: Discrete numerical values.  No missing data or outliers
origin: Encoded categorical variable representing continent of origin.  No missing values or outliers.
"""

# 8.  Removing cases with missing data
"""
The only attribute with missing values is the horsepower column, and I will
impute these missing values with the numerical mean.
"""

# 9/10. Imputing missing values and removing outliers
# The horsepower column is the only one with missing values, so I will
# impute the missing values ('?') with the numeric mean of the other
# values in that column.
Auto.loc[:, "horsepower"] = pd.to_numeric(Auto.loc[:, "horsepower"], errors='coerce')
HasNan = np.isnan(Auto.loc[:,"horsepower"])
Auto.loc[HasNan, "horsepower"] = np.nanmean(Auto.loc[:,"horsepower"])
Auto.loc[:, "horsepower"] = pd.to_numeric(Auto.loc[:, "horsepower"])

horsepower = Auto.loc[:,'horsepower']
flag = ((np.mean(horsepower) + 2 * np.std(horsepower)) < horsepower) & ((np.mean(horsepower) - 2 * np.std(horsepower)) > horsepower)
sum(flag)
# There are no outliers in this data set.  I noticed this after observing the distributions,
# but feel justified after checking the variable with the largest spread and observing
# zero outliers.

# 11. Decoding
# Decode the origin column, which represents the continent of origin.
NA_flag = Auto.loc[:, "origin"] == 1
EU_flag = Auto.loc[:, "origin"] == 2
AS_flag = Auto.loc[:, "origin"] == 3
Auto.loc[NA_flag, "origin"] = "North America"
Auto.loc[EU_flag, "origin"] = "Europe"
Auto.loc[AS_flag, "origin"] = "Asia"

# 12. Consolidation
# Consolidate the continuous numeric attribute into a categorical variable
# for weight. Low, medium, high with equal binning.
# Determine boundaries
weight = Auto.loc[:,"weight"]
bins = 3
BinWidth = (max(weight) - min(weight)) / bins
MinBin1 = float('-inf')
MaxBin1 = min(weight) + BinWidth
MaxBin2 = min(weight) + 2 * BinWidth
MaxBin3 = float('inf')

# Assign values to new bins.  Replace former weight column with new values.
eqBinnedWeight = np.empty(len(weight), object)
eqBinnedWeight[(MinBin1 < weight) & (weight <= MaxBin1)] = "Low Weight"
eqBinnedWeight[(MaxBin1 < weight) & (weight <= MaxBin2)] = "Med Weight"
eqBinnedWeight[(MaxBin2 < weight) & (weight <= MaxBin3)] = "High Weight"
Auto.loc[:, "binned_weight"] = eqBinnedWeight

# 13. One-hot encoding
# One-Hot-Encode the origin column into the columns North America, Europe, and Asia.
# Drop the now redundant column.
Auto.loc[:, "North America"] = (Auto.loc[:, "origin"] == "North America").astype(int)
Auto.loc[:, "Europe"] = (Auto.loc[:, "origin"] == "Europe").astype(int)
Auto.loc[:, "Asia"] = (Auto.loc[:, "origin"] == "Asia").astype(int)
Auto = Auto.drop("origin", axis=1)

# 14. Normalization
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

normAuto = normalize(Auto.loc[:, names[:-3]], names)

##############################################################################
# Unsupervised Learning
##############################################################################

# 1. Perform a K-Means with sklearn using some of your attributes.
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

# Plot horsepower vs cylinders and color by the cluster labels to see if
# the labels match up well with the cylinders variable.  There is some 
# ambiguity for 4,6 cylinder cars, but the 8 cylinder vehicles fit a
# cluster very well.
plt.scatter(Auto.loc[:,"horsepower"], Auto.loc[:,"cylinders"],c=labels)
plt.xlabel("Horsepower")
plt.ylabel("Cylinder")
plt.title("Horsepower vs Cylinder colored by labels")
plt.show()

# 2. Include at least one categorical column and one numeric attribute. 
# Neither may be a proxy for the expert label in supervised learning.

# Cylinder is the numerical attribute, and I will use cylinder as the categorical.

# 3. Normalize the attributes prior to K-Means or justify why you didn't normalize.

# Already normalized above! (Last step of data cleaning.)

# 4. Add the cluster label to the data set to be used in supervised learning
Auto.loc[:,"cluster_label"] = labels

##############################################################################
# Supervised Learning
##############################################################################

# 1. Ask a binary-choice question that describes your classification. Write the 
# question as a comment.
"""
Can a classifier predict whether a car was produced in Asia given the information
of a certain car like cyilinders, mpg, weight, etc.?

Answer: Yes with a random forrest classifier.
"""

# 2. Split your data set into training and testing sets using the proper 
# function in sklearn.
r = 0.3
TrainSet, TestSet = train_test_split(Auto, test_size=r)

# 3. Use sklearn to train two classifiers on your training set, like 
# logistic regression and random forest. 

# Predict whether a car was made in Asia
Target = "Asia"
Inputs = list(Auto.columns)
Inputs = Inputs[:-5] # Remove columns that cause leakage

# Create/fit random forest classifier
mss = 2 # mininum samples split parameter
estimitors = 10
clf1 = RandomForestClassifier(n_estimators=10, min_samples_split=mss) # default parameters are fine
clf1.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

# Naive Bayes classifier
nbc = GaussianNB()
nbc.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

# 4. Apply your (trained) classifiers to the test set.
# Random Forrest
BothProbabilities1 = clf1.predict_proba(TestSet.loc[:,Inputs])
probabilities1 = BothProbabilities1[:,1]

BothProbabilities2 = nbc.predict_proba(TestSet.loc[:,Inputs])
probabilities2 = BothProbabilities2[:,1]


# 5. Create and present a confusion matrix for each classifier. Specify and 
# justify your choice of probability threshold.

# I will set a threshold of 0.5 because I think that is a good point to maximize
# TPR and minimize FPR.
threshold = 0.5

predictions1 = (probabilities1 > threshold).astype(int)
CM1 = confusion_matrix(TestSet.loc[:,Target], predictions1)
tn1, fp1, fn1, tp1 = CM1.ravel()

predictions2 = (probabilities2 > threshold).astype(int)
CM2 = confusion_matrix(TestSet.loc[:,Target], predictions2)
tn2, fp2, fn2, tp2 = CM2.ravel()

# 6. For each classifier, create and present 2 accuracy metrics based on the 
# confusion matrix of the classifier.  I will use accuracy rate and f1 scores.
AR_rf = accuracy_score(TestSet.loc[:,Target], predictions1)
print("random forest accuracy score:", AR_rf)

f1_rf = f1_score(TestSet.loc[:,Target], predictions1)
print("random forest f1 score:", f1_rf)

precision_rf = precision_score(TestSet.loc[:,Target], predictions1)
print("random forest precision:", precision_rf)

AR_nbc = accuracy_score(TestSet.loc[:,Target], predictions2)
print("NBC accuracy score:", AR_nbc)

f1_nbc = f1_score(TestSet.loc[:,Target], predictions2)
print("NBC f1 score:", f1_nbc)

precision_nbc = precision_score(TestSet.loc[:,Target], predictions2)
print("NBC precision:", precision_nbc)

# 7. For each classifier, calculate the ROC curve and it's AUC using sklearn. 
# Present the ROC curve. Present the AUC in the ROC's plot.
# Generate True Positive Rate, Fale Positive Rate
fpr_rf, tpr_rf, th_rf = roc_curve(TestSet.loc[:,Target], probabilities1)
fpr_nbc, tpr_nbc, th_nbc = roc_curve(TestSet.loc[:,Target], probabilities2)

# Generate Area Under the ROC Curve
AUC_rf = auc(fpr_rf, tpr_rf)
print("Random Forrest AUC:", AUC_rf)
AUC_nbc = auc(fpr_nbc, tpr_nbc)
print("Naive Bayes Classifier AUC:", AUC_nbc)

plt.rcParams["figure.figsize"] = [8, 8] # Square
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr_rf, tpr_rf, LW=3, label='Random Forrest ROC curve (AUC = %0.2f)' % AUC_rf)
plt.plot(fpr_nbc, tpr_nbc, LW=3, label='Naive Bayes Classifier ROC curve (AUC = %0.2f)' % AUC_nbc)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()