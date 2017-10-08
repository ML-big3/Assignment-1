#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:14:27 2017

@author: xu

#Assignment-1 Task-1
"""

import config
import training
import pandas as pd
from sklearn.preprocessing import Imputer


house_dataset = pd.read_csv(config.HOUSE_DS)

feature_cols = ["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
                "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", 
                "1stFlrSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
                "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
                "OpenPorchSF", "EnclosedPorch", "MiscVal", "MoSold", "YrSold"]
result = 'SalePrice'

# Load Data
X = house_dataset[feature_cols].values
y = house_dataset[result].values

# Data mining
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,6].reshape(-1,1))
X[:,[6]] = imputer.transform(X[:,6].reshape(-1,1))
imputer = imputer.fit(X[:,21].reshape(-1,1))
X[:,[21]] = imputer.transform(X[:,21].reshape(-1,1))

# Perform the algorithms
training.linear_regression(X, y)
training.svr(X, y)