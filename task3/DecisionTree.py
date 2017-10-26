#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:24:47 2017

"""

import evaluation

from sklearn.tree import DecisionTreeClassifier

def decisionTreeClassifier(X, y) :
    '''
    Decision Tree Classifier
    '''    
    classifier = DecisionTreeClassifier(criterion = 'entropy')    
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, "DecisionTrees")
    
    evaluationMetric.crossValidateForAccuracy()
    evaluationMetric.crossValidatePrecisionScore()
    evaluationMetric.crossValidateLogLoss()
    evaluationMetric.crossValidateAucRoc()
    #evaluationMetric.crossValidateConfusionMatrix()
    evaluationMetric.crossValidateRecall()
    evaluationMetric.crossValidateF1()
    evaluationMetric.timeToTrain()
    evaluationMetric.trainingMemory()
