#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:05:04 2017

@author: amit
"""

"""
Assignment 1 - Task 1

Linear Regression

Implementing Linear Regression on the SUM data without noise

"""

import config
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Importing the dataset
dataset = pd.read_csv(filepath_or_buffer = config.SUM_WO_NOISE_DS, sep = ';')
y = dataset[['Target']].values
X = dataset[['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 6',
                        'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']].values
             
# Encoding y column
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y[:,:])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test, y_test)
             
