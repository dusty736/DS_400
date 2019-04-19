#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 00:15:49 2019

@author: dusty

Dustin Burnham
Data Science 400
2/12/2019
Milestone Project 2: Data Preparation
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read in the data from a freely available source on the internet.  
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'income']

census = pd.read_csv(url, header=None)

# Assign names
census.columns = names

# 2. Account for outlier values in numeric columns (at least 1 column).
# Replace outliers for age with the median.
def replace_outliers(df, col):
    """
    Input: dataframe
    Output: dataframe where the outliers have been replaced by the median of 
    that column.
    
    I will check to see which values meet the conditions that they fall
    between +- 2 standardard deviations of the means.  i will use the tilda
    to pick out the outliers and replace them with the median.
    """
    
    high = np.mean(df[col]) + 2 * np.std(df[col])
    low = np.mean(df[col]) - 2 * np.std(df[col])
    FlagGood = (df.loc[:, col] < high) & (df.loc[:, col] > low)
    df.loc[~FlagGood,col] = np.median(df[col])
    return(df)
    
census = replace_outliers(census, "age")

# 3. Replace missing numeric data (at least 1 column).
# Replace missing numeric data from the hours-per-week attribute
# with the median hours worked per week.
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
    
census = replace_outliers(census, "hours-per-week")

# After replacing missing numerical values in hours-per-week, I will remove the 
# remaining "?" values by replacing them with nan and then using the dropna
# function to remove all rows with any nan.
census = census.replace(" ?", float("nan"))
census = census.dropna(axis=0)

# 4. Normalize numeric values (at least 1 column, but be consistent with numeric data).
# I will normalize the age and hours column for plotting on a histogram.
# All values will be between 0 and 1.
def MinMaxNorm(df, col):
    """
    Input: dataframe name, column name
    Output: Array of normalized values using (x - max(x))/(max(x) - min(x))
    formula to feature scale the column.
    """
    
    col_tmp = df.loc[:, col]
    MinMax = (col_tmp - min(col_tmp)) / (max(col_tmp) - min(col_tmp))
    return(MinMax)
    
MinMaxAge = MinMaxNorm(census, 'age')
MinMaxHours = MinMaxNorm(census, 'hours-per-week')

census.loc[:, "min-max-age"] = MinMaxAge
census.loc[:, "min-max-hours"] = MinMaxHours

#plt.hist(MinMaxAge)
#plt.hist(MinMaxHours)

# 5. Bin numeric variables (at least 1 column).
# I will bin age into young, middle-aged, and senior buckets.  This
# will turn numerical data into categorical data.  Column will be replaced.
age = census.loc[:, 'age']

# Determine boundaries
bins = 3
BinWidth = (max(age) - min(age)) / bins
MinBin1 = float('-inf')
MaxBin1 = min(age) + BinWidth
MaxBin2 = min(age) + 2 * BinWidth
MaxBin3 = float('inf')

# Assign values to new bins.  Replace former age column with new values.
eqBinnedAge = np.empty(len(age), object)
eqBinnedAge[(MinBin1 < age) & (age <= MaxBin1)] = "young"
eqBinnedAge[(MaxBin1 < age) & (age <= MaxBin2)] = "middle-aged"
eqBinnedAge[(MaxBin2 < age) & (age <= MaxBin3)] = "senior"

census.loc[:, 'age'] = eqBinnedAge

# 6. Consolidate categorical data (at least 1 column).
# Consolidate marital-status into married or not married.
married = census.loc[:, 'marital-status']
MarriedOrNot = np.empty(len(married), object)
MarriedOrNot[married == " Married-civ-spouse"] = 'Married'
MarriedOrNot[married != " Married-civ-spouse"] = 'Not Married'
census.loc[:, 'marital-status'] = MarriedOrNot

# 7. One-hot encode categorical data with at least 3 categories (at least 1 column).
# One hot encode the race categorical variable, giving each race its own
# attribute of 1s and 0s for analsis.
census.loc[:, "White"] = (census.loc[:, "race"] == " White").astype(int)
census.loc[:, "Black"] = (census.loc[:, "race"] == " Black").astype(int)
census.loc[:, "Asian-Pac-Islander"] = (census.loc[:, "race"] == " Asian-Pac-Islander").astype(int)
census.loc[:, "Amer-Indian-Eskimo"] = (census.loc[:, "race"] == " Amer-Indian-Eskimo").astype(int)
census.loc[:, "Other"] = (census.loc[:, "race"] == " Other").astype(int)

# 8. Remove obsolete columns (race).  Other columns like marital status and
# age were overwritten.
census = census.drop("race", axis=1)

# Return the new dataframe as a csv in the current working directory
# with the following filename.
filename = 'DustinBurnham-M02-Dataset.csv'
census.to_csv(filename)