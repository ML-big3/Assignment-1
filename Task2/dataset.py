# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def skinNoSkinDataset(path_data_set):
    dataset = pd.read_csv(path_data_set, sep='\t')
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    X = dataset.iloc[:, 0:3].values
    #X = X[:10000]
    y = dataset.iloc[:,3].values
    #y = y[:10000]
    
    # data cleaning
    y = [yi -1 for yi in y]
    
    return (X, y)

def susyDataset(path_data_set):
    dataset = pd.read_csv(path_data_set, sep=',')
    X = dataset.iloc[:, 1:].values
    #X = X[:10000]
    y = dataset.iloc[:,0].values
    #y = y[:10000]
    
    # data cleaning
    y = [int(yi) for yi in y]
    
    return (X, y)