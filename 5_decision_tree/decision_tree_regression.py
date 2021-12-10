# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 20:21:58 2021

@author: kkakh
"""

#decision tree regression


#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataseta
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

#train the decision tree regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#predict new result
regressor.predict([[6.5]])

#visualise the decisiion tree regression model (Higher resolution )
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict([[6.5]]), color='blue')
plt.title('Truth OR Bluff Decision tree regression')
plt.xlabel('position label')
plt.ylabel('Salary')
plt.show
