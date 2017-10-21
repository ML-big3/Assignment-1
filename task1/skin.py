#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Assignment 1 - Task 1 

Linear Regression

"""

import config
import training
import pandas as pd
from sklearn.utils import shuffle
# Importing the dataset
data = pd.read_csv(config.SKIN_NO_SKIN, sep="\t", names=["a", "b", "c", "d"])
y = data["d"].values
X = data[["a", "b", "c"]].values
# training.linear_regression(X , y)

X, y = shuffle(X, y)

training.linear_regression(X, y)
training.svr(X, y)
training.logistic_regression(X, y)
training.svc(X, y)


