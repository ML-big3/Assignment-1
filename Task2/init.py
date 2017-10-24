# -*- coding: utf-8 -*-

import config
import dataset
import DecisionTree
import LogisticRegression
import SVM
import KNN


X, y = dataset.skinNoSkinDataset(config.SKIN_DATA_SET)
#X, y = dataset.susyDataset(config.SUSY_DATA_SET)


LogisticRegression.logisticRegressionClassifier(X, y)
SVM.svmClassifier(X, y)
DecisionTree.decisionTreeClassifier(X, y)
KNN.knnClassifier(X,y) 



