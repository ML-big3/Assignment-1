#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:34:27 2017

@author: xu

#Assignment-1 Task-1
"""

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics

def linear_regression(X_dataframe, y_dataframe, chunks):
    for i in chunks:
        if X_dataframe.size >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.2, random_state = 0)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def other_algorithm1():

	#code here

def other_algorithm2(data):
	#code here

def other_algorithm3:
	#code here