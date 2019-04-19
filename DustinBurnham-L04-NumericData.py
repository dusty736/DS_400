#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:27:28 2019

@author: dusty

Dustin Burnham
Data Science 400
Lesson 4 Assingment

Dataset: Heart disease data from Cleveland.  Obtained from the url below.

There were two types of outliers that were common in the dataset: missing values
and numerical outliers.  The missing values were denoted by a "?", and the numerical
outliers were values that were greater than 2 standard deviations from the mean
of their column.  The attributes that required imputations of missing values were
the "ca" and "thal" columns.  I needed to replace the missing values because the
type of the columns were objects instead of numerical, which made calculating 
statistical values impossible.  I replaced the "?"s in the two columns with
nan's and then those with the median of the non-missing numerical data.  
The attribute I histogrammed was age, so I could see the distribution of ages
of people in the study.  Unsurprisingly the age distribution is centered around 
age 55, so the age group is definitely older.  Only the "fbs" attribute was
removed because of the high amount of numerical outliers.  I did not remove
any rows since the attributes that contained missing values did not contain 
bad data in the other attributes.  Replacing these values with the median
allowed for preservation of that patients other information.
"""

# Load in the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collect Cleveland Heart Disease data from the url.  No header.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart = pd.read_csv(url, header=None)

# Assign column names
heart.columns = ['age', 'sex', 'cp-type', 'rest-bp', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Impute and assign median values to missing numeric values. (In "chol" and "thal)
def replace_median(df, col):
    """
    Input: dataframe, column where values will be replaced.  
    Output: dataframe with median values replacing the missing values.
    
    First we will coerce the column to be numeric, where all errors
    wll be replace with a nan.  Next we find the missing values using the
    np.isnan() function which returns a boolean array that picks out those values.
    Finally we replace those nan's with the median of the non-missing entries.
    """
    df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors='coerce')
    HasNan = np.isnan(df.loc[:,col])
    df.loc[HasNan, col] = np.nanmedian(df.loc[:, col])
    return(df)
    
heart = replace_median(heart, "ca")
heart = replace_median(heart, "thal")

# Check there are no more missing values.
#heart.isnull().sum()

# Replace outliers
def replace_outliers(df):
    """
    Input: dataframe
    Output: dataframe where the outliers have been replaced by the median of 
    that column.
    
    I will check to see which values meet the conditions that they fall
    between +- 2 standardard deviations of the means.  i will use the tilda
    to pick out the outliers and replace them with the median.
    """
    
    names = df.columns
    for col in names:
        high = np.mean(df[col]) + 2 * np.std(df[col])
        low = np.mean(df[col]) - 2 * np.std(df[col])
        FlagGood = (df.loc[:, col] < high) & (df.loc[:, col] > low)
        print(col, sum(~FlagGood))
        df.loc[~FlagGood,col] = np.median(df[col])
    return(df)

heart = replace_outliers(heart)

# Remove fbs attribute due to high number of numerical outliers.
heart = heart.drop(columns='fbs')

# Plot Histogram of age with outliers replaced
plt.hist(heart.loc[:, "age"])
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Histogram")
plt.show()

# Plot Scatter of resting blood pressure vs. age with the outliers replaced.
plt.scatter(heart["age"], heart["rest-bp"])
plt.xlabel("Age")
plt.ylabel("Resting Blood Pressure")
plt.title("Age vs. Resting Blood Pressure")
plt.show()

def calc_stdev(df):
    print("Standard Deviations")
    for i in df.columns:
        print(i, ":", np.std(df.loc[:, i]))
        
calc_stdev(heart)