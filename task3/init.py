# -*- coding: utf-8 -*-

import config
import functions
import numpy as np
import pandas as pd

def startTraining(X, y):
    functions.logisticRegressionClassifier(X, y)
    functions.svmClassifier(X, y)
    functions.decisionTreeClassifier(X, y)
    functions.knnClassifier(X,y) 

def trainingSusyDataset():
    dataset = pd.read_csv(config.SUSY_DATA_SET, sep=',')
    X = dataset.iloc[:, 1:].values
    X = X[:10000]
    y = dataset.iloc[:,0].values
    y = y[:10000]
    
    # data cleaning
    y = [int(yi) for yi in y]

    startTraining(X, y)


def trainingSkinNoSkinDataset():
    dataset = pd.read_csv(config.SKIN_DATA_SET, sep='\t')
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    X = dataset.iloc[:, 0:3].values
    X = X[:10000]
    y = dataset.iloc[:,3].values
    y = y[:10000]
    
    # data cleaning
    y = [yi -1 for yi in y]

    startTraining(X, y)


trainingSusyDataset()
trainingSkinNoSkinDataset()