#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:34:27 2017

@author: xu

#Assignment-1 Task-1
"""
CHUNKS = [100, 500, 1000, 5000, 10000]
#CHUNKS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
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

            #RMSE metric
            print(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()))

            #R2 metric
            print(cross_val_score(regressor, X, y, cv=10, scoring="r2").mean())

def decision_tree_regression(X_dataframe, y_dataframe):
    print("Decision Tree Regression")
    for i in CHUNKS:
        rmse = 0
        r2 = 0
        if X_dataframe.shape[0] >= i:
            X = X_dataframe[:i]
            y = y_dataframe[:i]
            regressor = DecisionTreeRegressor(max_depth=5)

            #RMSE metric
            print(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()))

            #R2 metric
            print(cross_val_score(regressor, X, y, cv=10, scoring="r2").mean())

def logistic_regression(X_dataframe, y_dataframe):
    print("LogisticRegression")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X = X_dataframe[:i]
            y = y_dataframe[:i]
            clf = LogisticRegression()

            #Accuracy metric
            print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())

            #f1 score
            print("f1 ", cross_val_score(clf, X, y, cv=10, scoring="f1").mean())


            #f1 metric
            # y_label = list(set(y))
            # if len(y_label) > 2:
            #     scoring = metrics.make_scorer(metrics.f1_score, labels = y_label, average = "weighted")
            # else:
            #     scoring = metrics.make_scorer(metrics.f1_score)
            # print("v_measure_score  ", cross_val_score(clf, X, y, cv=10, scoring=scoring).mean())

def logistic_regression_with_split_metrics(X_dataframe, y_dataframe):
    print("logistic_regression")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            
            #Accuracy metric            
            print("accuracy", clf.score(X_test, y_test))

            #f1 Score
            y_pred = clf.predict(X_test)
            y_label = list(set(y_test))
            print("f1 score", metrics.f1_score(y_test, y_pred, labels=y_label, average='weighted'))

def knn(X_dataframe, y_dataframe):
    print("KNN")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X = X_dataframe[:i]
            y = y_dataframe[:i]
            clf = KNeighborsClassifier(10, weights="uniform")

            # Accuracy metric
            print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())

            # f1 score
            print("f1 ", cross_val_score(clf, X, y, cv=10, scoring="f1").mean())

def knn_with_split_metrics(X_dataframe, y_dataframe):
    print("KNN")
    for i in CHUNKS:
        if X_dataframe.shape[0] >= i:
            X_train, X_test, y_train, y_test = train_test_split(X_dataframe[:i], y_dataframe[:i], test_size = 0.3, random_state = 0)
            clf = KNeighborsClassifier(10, weights="uniform")
            clf.fit(X_train, y_train)
            
            #Accuracy metric
            print("accuracy", clf.score(X_test, y_test))

            #f1 Score
            y_pred = clf.predict(X_test)
            y_label = list(set(y_test))
            print("f1 score", metrics.f1_score(y_test, y_pred, labels=y_label, average='weighted'))