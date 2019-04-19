#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:31:49 2019

@author: dusty
"""

import pandas as pd
Vehicle = pd.DataFrame()
Vehicle.loc[:,"Type"] = ["Tricycle", "Car", "Motorcycle"]
Wheels = [3, 4, 2]
Vehicle.loc[:,"Wheels"] = Wheels
Vehicle.loc[3, :] = ["Sled", 0]
Vehicle.loc[:, "Doors"]  = [0, 2, 0, 0]

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
Iris = pd.read_csv(url, header=None)
Iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

plt.hist(Iris.loc[:, "petal_length"])

#_ = scatter_matrix(Mamm, c=Mamm.loc[:,"Severity"], figsize=[8,8], s=1000)
_ = scatter_matrix(Iris)