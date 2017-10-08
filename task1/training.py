#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:34:27 2017

@author: xu

#Assignment-1 Task-1
"""
CHUNKS = [100, 500, 1000, 5000]
#CHUNKS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
from sklearn import svm

# Linear Regression Algorithm
def linear_regression(X_dataframe, y_dataframe):
    print("LinearRegression")
    for i in CHUNKS:
        if X_dataframe.size >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Support Vector Regression Algorithm
def svr(X_dataframe, y_dataframe):
    print("SVR")
    for i in CHUNKS:
        if X_dataframe.size >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            clf = svm.SVR()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#def other_algorithm2(data):
	#code here

#def other_algorithm3:
	#code here