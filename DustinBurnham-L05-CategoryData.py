#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:20:45 2019

@author: dusty

Dustin Burnham
2/5/19
Data Science 400
Assignment 5:Categorical Data

Summary:
    The only categorical attributes in the Auto data is the "origin" column.
    The values are numerical, so I will decode the column.  Next I will replace
    the missing values with an "unknown" value.  Because this is the only 
    categorical column, I will consolidate these values into a new column
    that determines whether the car is foreign or domestic.  Finally I will
    one-hot encode the attribute for numerical analysis and drop the original  
    column.  I also turn the weight attribute from numerical to categorical by 
    using equal-width binning.  This will be helpful for plotting using a bar
    plot.  We can see that the low weight cars are the most common using this
    type of plot.
"""

# 1.  Import statements for necessary packages
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2.  Read in the dataset with car data from the url below.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
Auto.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]

# 3.  Normalize numeric values (at least 1 column, but be consistent with other numeric data).
# I normalize the horsepower and  weight attributes using the min max normalization.
# Horsepower has missing values, so I will replace them with the mean of the non-missing values.
Auto.loc[:, "horsepower"] = pd.to_numeric(Auto.loc[:, "horsepower"], errors='coerce')
HasNan = np.isnan(Auto.loc[:,"horsepower"])
Auto.loc[HasNan, "horsepower"] = np.nanmean(Auto.loc[:,"horsepower"])
Auto.loc[:, "horsepower"] = pd.to_numeric(Auto.loc[:, "horsepower"])

hp = Auto.loc[:, "horsepower"]
weight = Auto.loc[:, "weight"]

# Normalize the attributes.
hpMinMax = (hp - min(hp)) / (max(hp) - min(hp))

hpMinMax = (weight - min(weight)) / (max(weight) - min(weight))

# 4.  Bin numeric variables (at least 1 column).  Equal width bins for the weight
# Attribute.  The bins will be split into low, medium, and high weight.

# Determine boundaries
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
Auto.loc[:, "weight"] = eqBinnedWeight

# 5. Decode categorical data.  Continent of origin.  "origin" column.
Replace = Auto.loc[:, "origin"] == 1
Auto.loc[Replace, "origin"] = "North America"

Replace = Auto.loc[:, "origin"] == 2
Auto.loc[Replace, "origin"] = "Europe"

Replace = Auto.loc[:, "origin"] == 3
Auto.loc[Replace, "origin"] = "Asia"

# 6.  Impute missing categories.
# Replace missing values ("?") with "unknown" label.
MissingValue = Auto.loc[:, "origin"] == "?"
Auto.loc[MissingValue, "origin"] = "unknown"

# 7.  Consolidate categorical data (at least 1 column).
# The origin attribute seemed natural for consolidation.  I consolidated the
# values for European and Asian cars to the "Foreign" value, while "Domestic"
# for North American.
origin = Auto.loc[:,"origin"]
foreignDomestic = np.empty(len(origin), object)
foreignDomestic[origin == "North America"] = "Domestic"
foreignDomestic[(origin == "Asia") | (origin == "Europe")] = "Foreign"
Auto.loc[:,"foreign/Domestic"] = foreignDomestic

# 8.  One-hot encode categorical data with at least 3 categories (at least 1 column).
# I will used the origin attribute again to hot encode the country of origin.
# The three new columns will be for the three possible values North America,
# Europe, and Asia where each value is a zero or one.
Auto.loc[:, "North America"] = (Auto.loc[:, "origin"] == "North America").astype(int)
Auto.loc[:, "Europe"] = (Auto.loc[:, "origin"] == "Europe").astype(int)
Auto.loc[:, "Asia"] = (Auto.loc[:, "origin"] == "Asia").astype(int)

# 9.  Remove obsolete columns.  I've one-hot encoded the origin attribute,
# so we don't need it anymore and will drop it from the Auto dataframe.
Auto = Auto.drop("origin", axis=1)

# 10.  Present plots for 1 or 2 categorical columns.  I will plot the weight
# attribute because I turned that from numerical to categorical.  A bar plot
# shows the distribution well.
Auto.loc[:,"weight"].value_counts().plot(kind='bar')