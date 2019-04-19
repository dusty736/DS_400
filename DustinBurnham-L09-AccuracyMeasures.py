#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dustin Burnham
Data Science 400
3/5/19
Assignment 9: Accuracy Measures

Summary:
    The goal was to classify/predict whether a car was made in Asia based on
    other factors like mpg, cylinders, horsepower, etc..  I decided since the
    asia column was one-hot-encoded, that I would use a random forrest classifier
    to make the predictions of the host nation.  Once I trained my model and 
    created my probabilities, I used a threshold of 0.5 to create predictions.
    Next I calculated the TPR, FPR, precision, recall, accuracy score, ROC
    curve, and finally the AUC of that ROC curve.
    Results (These numbers slightly vary after each run):
        Accuracy Score: 0.82
        Precision: 0.72
        Recall: 0.43
        f1 score: 0.54
        Area Under Curve: 0.90
    
    My model predicted whether a vehicle was made in Asia quite well with an
    accuracy score of 0.82 and an area under the curve 0f 0.9.  
"""

##############################################################################
# Import statements for necessary packages
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib

##############################################################################
# Read in the dataset from a freely and easily available source on the internet.
##############################################################################

# Read in data, add header, drop useless column
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
names = ["mpg", "cylinders", "displacement", "horsepower", 
                "weight", "acceleration", "model_year", "origin", "car_name"]
Auto.columns = names
Auto = Auto.drop("car_name", axis=1)

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

##############################################################################
# Split your data set into training and testing sets using the proper function 
# in sklearn (include decision comments).
##############################################################################

# 70% training data, 30% test data
r = 0.3
TrainSet, TestSet = train_test_split(Auto, test_size=r)

# Predict whether a car was made in Asia
Target = "Asia"
Inputs = list(Auto.columns)
Inputs = Inputs[:-3] # Remove columns that cause leakage

# Create/fit random forest classifier
mss = 2 # mininum samples split parameter
estimitors = 10
clf = RandomForestClassifier(n_estimators=10, min_samples_split=mss) # default parameters are fine
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

# Generate probabilites for each prediction using predict_proba function
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]

##############################################################################
# 1.  Confusion Matrix
##############################################################################

# I will set a threshold of 0.5 because I think that is a good point to maximize
# TPR and minimize FPR.
threshold = 0.5

# Generate predictions, confusion matrix
predictions = (probabilities > threshold).astype(int)
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()

##############################################################################
# 2.  Precision, Recall, f1 score
##############################################################################

AR = accuracy_score(TestSet.loc[:,Target], predictions)
print("accuracy score:", AR)
precision = precision_score(TestSet.loc[:,Target], predictions)
print("precision:", precision)
recall = recall_score(TestSet.loc[:,Target], predictions)
print("recall:", recall)
f1 = f1_score(TestSet.loc[:,Target], predictions)
print("f1 score:", f1)

##############################################################################
# 3.  ROC Curve, Area Under Curve (AUC) Score, and Plot
##############################################################################

# Generate True Positive Rate, Fale Positive Rate
fpr, tpr, th = roc_curve(TestSet.loc[:,Target], probabilities)

# Generate Area Under the ROC Curve
AUC = auc(fpr, tpr)
print("Area Under Curve:", AUC)

# Generate ROC Plot
plt.rcParams["figure.figsize"] = [8, 8] # Square
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()