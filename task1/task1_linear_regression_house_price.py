#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:14:27 2017

@author: zen

#Assignment-1 Task-1
"""

import config
import training
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold


house_dataset = pd.read_csv(config.HOUSE_DS)

#feature_cols = ["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", 
#                "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF",
#                "1stFlrSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
#                "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
#                "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
#                "OpenPorchSF", "EnclosedPorch", "MoSold", "YrSold" ]
feature_cols = ["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt"]
result = 'SalePrice'

#
X = house_dataset[feature_cols].values
y = house_dataset[result].values

# Perform the regresion
training.linear_regression(X,y, config.CHUNKS)
