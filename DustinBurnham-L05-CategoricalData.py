#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 19:56:54 2019

@author: dusty
"""

import numpy as np
import pandas as pd

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
mamm = pd.read_csv(url, header=None)
mamm.columns = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

minX = min(X)
maxX = max(X)

minMaxX = (X - minX)/(maxX - minX)
minMaxX = (X - np.min(X))/(np.max(X) - np.min(X))
np.mean(minMaxX)

import numpy as np
x = np.array(["WA", "Washington", "Wash", "UT", "Utah", "Utah", "UT", "Utah", "IO"])

x[x == "Washington"] = "WA"
x[x == "Wash"] = "WA"
x[x == "Utah"] = "UT"