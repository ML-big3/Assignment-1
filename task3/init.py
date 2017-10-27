# -*- coding: utf-8 -*-

import config
import dataset
import DecisionTree
import LogisticRegression
import SVM
import KNN

def start_training(X, y):
    LogisticRegression.logistic_regression_classifier(X, y)
    DecisionTree.decision_tree_classifier(X, y)
    KNN.knn_classifier(X,y)
    SVM.svm_classifier(X, y)

X, y = dataset.skin_noskin_dataset(config.SKIN_DATA_SET)
start_training(X, y)

X, y = dataset.susy_dataset(config.SUSY_DATA_SET)
start_training(X, y)