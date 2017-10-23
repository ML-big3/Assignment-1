#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:24:47 2017

"""

import numpy as np
import pandas as pd

import config
import evaluation


dataset = pd.read_csv(config.SKIN_DATA_SET, sep='\t')
dataset = dataset.reindex(np.random.permutation(dataset.index))
X = dataset.iloc[:, 0:3].values
X = X[:10000]
y = dataset.iloc[:,3].values
y = y[:10000]

# data cleaning
y = [yi -1 for yi in y]


# Decision Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')


# Evaluating the performance using 10 fold cross validation
evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7)

evaluationMetric.crossValidateForAccuracy()
evaluationMetric.crossValidatePrecisionScore()
evaluationMetric.crossValidateLogLoss()
evaluationMetric.crossValidateAucRoc()
evaluationMetric.crossValidateConfusionMatrix()
