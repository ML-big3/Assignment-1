import numpy as np
import pandas as pd
import evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def decisionTreeClassifier(X, y) :
    '''
    Decision Tree Classifier
    '''    
    classifier = DecisionTreeClassifier(criterion = 'entropy')    
    
    # Evaluating the performance using 10 fold cross validation
    metrics(classifier, 'Decision Trees', X, y)

def knnClassifier(X, y):
    """
    KNN
    """
    
    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = KNeighborsClassifier(n_neighbors= 20, metric = 'minkowski', p=2)
    
    # Evaluating the performance using 10 fold cross validation
    metrics(classifier, 'KNN', X, y)



def logisticRegressionClassifier(X, y):
    """
    Logistic Regression
    """
    
    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = LogisticRegression(random_state = 0)

    # Evaluating the performance using 10 fold cross validation
    metrics(classifier, 'logisticRegression', X, y)

def svmClassifier(X, y):
    """
    SVM
    """
    
    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = SVC(kernel = 'linear', random_state = 0)
    
    # Evaluating the performance using 10 fold cross validation
    metrics(classifier, 'SVM', X, y)

def metrics(classifier, name, X, y):
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, name)
    evaluationMetric.crossValidateForAccuracy()
    evaluationMetric.crossValidatePrecisionScore()
    evaluationMetric.crossValidateLogLoss()
    evaluationMetric.crossValidateAucRoc()
    evaluationMetric.crossValidateConfusionMatrix()

    