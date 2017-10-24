#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:09:37 2017

Evaluation Metrics
"""

from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

class EvaluationMetrics:
    
    def __init__(self, classifier, X, y, splits, seed, classifier_name):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.kfold = model_selection.KFold(n_splits=10, random_state=seed)
        self.classifier_name = classifier_name
        
    def crossValidateForAccuracy(self):
        results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring='accuracy')
        print("Classifier "+self.classifier_name+" - Accuracy: %.3f (%.3f)") % (results.mean(), results.std())

    def crossValidateLogLoss(self):
        scorer = make_scorer(log_loss)
        results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring=scorer)
        print("Classifier "+self.classifier_name+" - LogLoss: %.3f (%.3f)") % (results.mean(), results.std())

    def crossValidateAucRoc(self):
        scorer = make_scorer(roc_auc_score)
        results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring=scorer)
        print("Classifier "+self.classifier_name+" - Area Under ROC Curve: %.3f (%.3f)") % (results.mean(), results.std())        

    def crossValidatePrecisionScore(self):
        scorer = make_scorer(precision_score)
        results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring=scorer)
        print("Classifier "+self.classifier_name+" - Precision Score: %.3f (%.3f)") % (results.mean(), results.std())
    
    def crossValidateConfusionMatrix(self):
        def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
        def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
        def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
        def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
        cm = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
                   'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}
        results = model_selection.cross_validate(self.classifier, self.X, self.y, cv=self.kfold, scoring=cm)
        print("Classifier "+self.classifier_name+" - Confusion Maxtrix: ", results)
        
