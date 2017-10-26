# Support Vector Machine (SVM)

# Importing the libraries



import evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def svmClassifier(X, y):
    """
    SVM
    """
    
    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = SVC(kernel = 'linear', random_state = 0)
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, "SVM")
    
    evaluationMetric.crossValidateForAccuracy()
    evaluationMetric.crossValidatePrecisionScore()
    evaluationMetric.crossValidateLogLoss()
    evaluationMetric.crossValidateAucRoc()
    #evaluationMetric.crossValidateConfusionMatrix()
    evaluationMetric.crossValidateRecall()
    evaluationMetric.timeToTrain()
    evaluationMetric.trainingMemory()
