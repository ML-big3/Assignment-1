#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 01:28:00 2017

Logistic Regression 
"""

import evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def logisticRegressionClassifier(X, y):
    """
    Logistic Regression
    """
    
    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = LogisticRegression(random_state = 0)
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, "LogisticRegression")
    
    evaluationMetric.crossValidateForAccuracy()
    evaluationMetric.crossValidatePrecisionScore()
    evaluationMetric.crossValidateLogLoss()
    evaluationMetric.crossValidateAucRoc()
    #evaluationMetric.crossValidateConfusionMatrix()
    evaluationMetric.crossValidateRecall()
    evaluationMetric.crossValidateF1()
    evaluationMetric.timeToTrain()
    evaluationMetric.trainingMemory()