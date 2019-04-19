#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:48:55 2019
Author: Dustin Burnham
Lesson 3 Assignment

Description:
In this script I will generate arrays that contain both numeric outliers
values of the incorrect type.  The goal will be to clean these arrays for
these outlier values by removal, replacing with a mean, and replacing with
the median.  The cleaned arrays will be returned once the script is run.
"""

# Import numpy to work with numpy arrays
import numpy as np

# Generate an array of 30 integers using the numpy.random.randint
arr1 = np.random.randint(1, 10, 30)

# Add in outliers at posiitons 5, 10, 15
arr1[[5, 10, 15]] = [100, 120, 150]

# Create a numpy array with numbers and characters that should not be there.
arr2 = np.array([1, 2, "!", 4, 8, "?", "", 3, 5])

def remove_outlier(arr):
    """
    Remove outliers from the input numpy array arr.
    Outliers are values greater than two standard deviations from the mean.
    """
    
    high = np.mean(arr) + 2 * np.std(arr)
    low = np.mean(arr) - 2 * np.std(arr)
    good_flag = (arr >= low) & (arr <= high)
    return(arr[good_flag])
    
def replace_outlier(arr):
    """
    Replace the outliers of the numpy array with the mean of the 
    non-outliers.  Return the numpy array with the new values.
    """
    
    high = np.mean(arr) + 2 * np.std(arr)
    low = np.mean(arr) - 2 * np.std(arr)
    good_flag = (arr >= low) & (arr <= high)
    mean = np.mean(arr[good_flag])
    arr[~good_flag] = mean
    return(arr)
    
def fill_median(arr):
    """
    The input numpy array, arr, contains numeric values and non-numeric
    values.  We will filter for these non-numeric values and replace them
    with the median of the numeric values of arr.  arr is returned.
    """
    
    good_flag = np.array([i.isdigit() for i in arr])
    int_arr = [int(i) for i in arr[good_flag]]
    med = np.median(int_arr)
    arr[~good_flag] = med
    arr = [float(i) for i in arr]
    return(arr)

# Call the three functions to clean arr1 and arr2
remove_outlier(arr1)
replace_outlier(arr1)
fill_median(arr2)