# AnscombeQuartet.py
# Copyright 2018 by Ernst Henle

# Anscombe's Quartet
# Four different datasets have the following measures
# Mean of x:  9
# Sample variance of x: 11
# Mean of y:	7.50
# Sample variance of y:  4.125
# Correlation between x and y:  0.816
# Linear regression line:  y = 3.00 + 0.500x

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Anscombe's Quartet Data
x0 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y0 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x3 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y3 = [6.58, 5.76, 7.71, 8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89]

lx = np.array([0., 20.])
ly = np.array([3., 13.])

plt.figure(1, facecolor='lightgrey')

plt.subplot(221)
plt.plot(x0, y0, 'bo', ms=8, mfc='none')
plt.xlim(0,20)
plt.ylim(0,14)
plt.text(x=2,y=12,s="x0, y0")
plt.plot([0., 20.], [3., 13.], '--k')

plt.subplot(222)
plt.plot(x1, y1, 'bo', ms=8, mfc='none')
plt.xlim(0,20)
plt.ylim(0,14)
plt.text(x=2,y=12,s="x1, y1")
plt.plot([0., 20.], [3., 13.], '--k')

plt.subplot(223)
plt.plot(x2, y2, 'bo', ms=8, mfc='none')
plt.xlim(0,20)
plt.ylim(0,14)
plt.text(x=2,y=12,s="x2, y2")
plt.plot([0., 20.], [3., 13.], '--k')

plt.subplot(224)
plt.plot(x3, y3, 'bo', ms=8, mfc='none')
plt.xlim(0,20)
plt.ylim(0,14)
plt.text(x=2,y=12,s="x3, y3")
plt.plot([0., 20.], [3., 13.], '--k')

plt.show()

def DataStats(x, y):
    RowOfStats = np.empty((7,))
    RowOfStats[0] = np.mean(x)
    RowOfStats[1] = np.mean(y)
    RowOfStats[2] = np.var(x, ddof=1)
    RowOfStats[3] = np.var(y, ddof=1)
    RowOfStats[4] = np.corrcoef(x, y)[0,1]
    regr = LinearRegression()
    regr.fit(np.array(x, ndmin=2).T, y)
    RowOfStats[5] = regr.intercept_
    RowOfStats[6] = regr.coef_[0]
    return RowOfStats

TableOfStats = pd.DataFrame(columns = ["Mean(X)", "Mean(Y)", "Var(X)", 
                                       "Var(Y)", "Corr", "Intercept", "Slope"])
TableOfStats.loc[0,:] = DataStats(x0, y0).round(2)
TableOfStats.loc[1,:] = DataStats(x1, y1).round(2)
TableOfStats.loc[2,:] = DataStats(x2, y2).round(2)
TableOfStats.loc[3,:] = DataStats(x3, y3).round(2)

print(TableOfStats)
