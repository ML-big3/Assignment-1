#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:34:27 2017

@author: xu

#Assignment-1 Task-1
"""
# CHUNKS = [100, 500, 1000, 5000, 10000]
CHUNKS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
# Linear Regression Algorithm
def linear_regression(X_dataframe, y_dataframe):
    print("LinearRegression")
    for i in CHUNKS:
        rmse = 0
        r2 = 0
        if X_dataframe.shape[0] >= i:
            X = X_dataframe[:i]
            y = y_dataframe[:i]
            regressor = LinearRegression()
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                rmse += np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                r2 += metrics.r2_score(y_test, y_pred)
            print(i, rmse/10)
            print(i, r2/10)

# Support Vector Regression Algorithm
def svr(X_dataframe, y_dataframe):
    print("SVR")
    for i in CHUNKS:
        rmse = 0
        r2 = 0
        if X_dataframe.shape[0] >= i:
            X = X_dataframe[:i]
            y = y_dataframe[:i]
            clf = svm.SVR()
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                rmse += np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                r2 += metrics.r2_score(y_test, y_pred)
            print(i, rmse/10)
            print(i, r2/10)

def logistic_regression(X_dataframe, y_dataframe):
    print("LogisticRegression")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            lr = LogisticRegression()
            lr.fit(X_train, y_train)

            #Mean accuracy on the test data and labels.
            print(lr.score(X_test, y_test))

def svc(X_dataframe, y_dataframe):
    print("SVC")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            clf = svm.SVC()
            clf.fit(X_train, y_train)

            #Mean accuracy on the test data and labels.
            print(clf.score(X_test, y_test))

def knn(X_dataframe, y_dataframe):
    print("KNN")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            # for weights in ['uniform', 'distance']
            clf = KNeighborsClassifier(10, weights="uniform")
            clf.fit(X_train, y_train)

            #Mean accuracy on the test data and labels.
            print(clf.score(X_test, y_test))

def decision_tree_regression(X_dataframe, y_dataframe):
    print("Decision Tree Regression")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            clf = DecisionTreeRegressor(max_depth=4)
            clf.fit(X_train, y_train)
            #Mean accuracy on the test data and labels.
            print(clf.score(X_test, y_test))

