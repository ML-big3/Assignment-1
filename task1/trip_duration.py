#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Assignment 1 - Task 1 

Linear Regression

"""

import config
import training
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(config.Trip_Duration, sep=",")
y = dataset['trip_duration'].values
# y = dataset['vendor_id'].values
# X = dataset[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude', 'vendor_id', 'passenger_count']].values
X = dataset[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude', 'passenger_count']].values

training.linear_regression(X, y)
training.logistic_regression(X, y)
# training.svc(X, y)



