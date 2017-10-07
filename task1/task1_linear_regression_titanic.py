#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Assignment 1 - Task 1 

Linear Regression

"""

import config
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv(config.TITANIC_DS)
y_dataframe = dataset[['Survived']]
X__dataframe = dataset[['Pclass','Sex','Age','SibSp','Parch','Fare']]

# Preprocessing

# extracting the data as an array
y = y_dataframe.values
X = X__dataframe.values


# handling missing values
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,2].reshape(-1,1))
X[:,[2]] = imputer.transform(X[:,2].reshape(-1,1))

# encoding the sex column
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Perform the regresion
training.linear_regression(X,y)



