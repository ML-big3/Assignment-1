#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:05:04 2017

@author: Ashish
"""

"""
Assignment 1 - Task 1

Linear Regression

Implementing Linear Regression on the SUM data with noise

"""

import config
import pandas as pd
import training

# Importing the dataset
dataset = pd.read_csv(filepath_or_buffer = config.SUM_WI_NOISE_DS, sep = ';')
y = dataset['Noisy Target'].values
X = dataset[['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless)','Feature 6',
                        'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']].values

# Perform the regresion
training.linear_regression(X , y)
training.svr(X, y)