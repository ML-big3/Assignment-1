# -*- coding: utf-8 -*-

import config
import numpy as np
import pandas as pd
import gzip
import shutil



def skin_noskin_dataset(path_data_set):
    dataset = pd.read_csv(path_data_set, sep='\t')
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    X = dataset.iloc[:, 0:3].values
    #X = X[:10000]
    y = dataset.iloc[:,3].values
    #y = y[:10000]
    
    # data cleaning
    y = [yi -1 for yi in y]
    
    return (X, y)

def susy_dataset(path_to_zip):
    # Extract zip in temporary folder

    with gzip.open(path_to_zip, 'rb') as f_in, open(config.TEMPORARY_DIR+"\SUSY.csv", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    dataset = pd.read_csv(config.TEMPORARY_DIR+"\SUSY.csv", sep=',')
    
    X = dataset.iloc[:, 1:].values
    X = X[:500]
    y = dataset.iloc[:,0].values
    y = y[:500]
    
    # data cleaning
    y = [int(yi) for yi in y]
    
    return (X, y)