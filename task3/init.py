# -*- coding: utf-8 -*-

import config
import dataset
import DecisionTree
import LogisticRegression
import SVM
import KNN


X, y = dataset.skin_noskin_dataset(config.SKIN_DATA_SET)
#X, y = dataset.susy_dataset(config.SUSY_DATA_SET)


LogisticRegression.logistic_regression_classifier(X, y)
DecisionTree.decision_tree_classifier(X, y)
KNN.knn_classifier(X,y)
SVM.svm_classifier(X, y)