# -*- coding: utf-8 -*-

import config
import dataset
import DecisionTree
import KNN


X, y = dataset.skinNoSkinDataset(config.SKIN_DATA_SET)

DecisionTree.decisionTreeClassifier(X, y)
KNN.knnClassifier(X,y) 

