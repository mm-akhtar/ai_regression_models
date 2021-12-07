# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:37:18 2021

@author: kkakh
"""
#Multiple linear regression 

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datsset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print(X)

#encoding the categorial data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

#split the datset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Training the multiple linear regression modal on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#predicting the test set results
Y_pred = regressor.predict(X_test)

#compare between actual and predicted values
np.set_printoptions(precision = 2)
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1) ), 1))

