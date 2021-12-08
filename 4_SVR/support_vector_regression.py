# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:07:12 2021

@author: kkakh
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#print X
print(X)

#print Y
print(y)

y = y.reshape(len(y),1)

#print Y after making it 2D
print(y)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
y = sc_Y.fit_transform(y)

#print X after feature scaling
print(X)

#print Y after feature scaling
print(y)

#train the SVR model on the whole database
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

#predict an new result
sc_Y.inverse_transform( regressor.predict(sc_X.transform([[6.5]])))

#visualise the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue')
plt.title('Truth OR Bluff (SVR)')
plt.xlabel('position lavel')
plt.ylabel('Salary')
plt.show()

#Visualise the SVR results with high resolution
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR Smooth)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()