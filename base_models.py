import numpy as np
from sklearn.linear_model import LinearRegression as SkLinReg, SGDClassifier as SkSGD, \
    Lasso as SkLasso, Ridge as SkRidge
from sklearn.svm import LinearSVC as SkSVC, LinearSVR as SkSVR
from sklearn.neural_network import MLPClassifier as SkMLPC, MLPRegressor as SkMLPR
from sklearn.tree import DecisionTreeClassifier as SkDecTreeC, DecisionTreeRegressor as SkDecTreeR

from ensemble import Ensemble


class Average(Ensemble):
    def __init__(self, result_type):
        self.result_type = result_type
        self.average = None
    
    def _fit(self, x, y, **kwargs):
        self.average = np.mean(y, axis=0)

    def predict(self, x):
        return np.array([self.average for _ in x])

class SkLearnModel(Ensemble):
    def _fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict(x)

class LogRegression(SkLearnModel):
    """
    Logistic regression 

    Wrapper of sklearn.linear_model.LogisticRegression
    """
    def __init__(self, **kwargs):
        self.model = SkSGD(loss="log", max_iter=1e2, **kwargs)
        self.result_type = 'classification'
    
    def _fit(self, x, y, **kwargs):
        self.model.fit(x, np.argmax(y, axis=1), **kwargs)
        print("DID NOT ESCAPE")
    
    def predict(self, x):
        return self.model.predict_proba(x)
    
class LinRegression(SkLearnModel):
    """
    Linear regresssion

    Wrapper of sklearn.linear_model.LinearRegression
    """
    def __init__(self, **kwargs):
        self.model = SkLinReg(**kwargs)
        self.result_type = 'regression'

class LassoRegressor(SkLearnModel):
    """
    Linear model with L1 regularization

    Wrapper of sklearn.linear_model.Lasso
    """
    def __init__(self, **kwargs):
        self.model = SkLasso(**kwargs)
        self.result_type = 'regression'

class RidgeRegressor(SkLearnModel):
    """
    Linear model with L2 regularization

    Wrapper of sklearn.linear_model.Ridge
    """
    def __init__(self, **kwargs):
        self.model = SkRidge(**kwargs)
        self.result_type = 'regression'

class LinearSVMClassifier(SkLearnModel):
    """
    Support vector machine for classification

    Wrapper of sklearn.svm.LinearSVC
    """
    def __init__(self, **kwargs):
        self.model = SkSVC(**kwargs)
        self.result_type = 'classification'
    
class LinearSVMRegressor(SkLearnModel):
    """
    Support vector machine for regression

    Wrapper of sklearn.svm.LinearSVR
    """
    def __init__(self, **kwargs):
        self.model = SkSVR(**kwargs)
        self.result_type = 'regression'
    
class MLPClassifier(SkLearnModel):
    """
    Multi-layer perceptron for classification

    Wrapper of sklearn.neural_network.MLPClassification
    """
    def __init__(self, **kwargs):
        self.model = SkMLPC(**kwargs)
        self.result_type = 'classification'
    
class MLPRegressor(SkLearnModel):
    """
    Multi-layer perceptron for regression

    Wrapper of sklearn.neural_network.MLPRegressor
    """
    def __init__(self, **kwargs):
        self.model = SkMLPR(**kwargs)
        self.result_type = 'regression'
    
class DecisionTreeClassifier(SkLearnModel):
    """
    Decision tree for classification

    Wrapper of sklearn.tree.DecisionTreeClassifier
    """
    def __init__(self, **kwargs):
        self.model = SkDecTreeC(**kwargs)
        self.result_type = 'classification'
    
    def _fit(self, x, y, **kwargs):
        self.model.fit(x, np.argmax(y, axis=1), **kwargs)

    def predict(self, x):
        return self.model.predict_proba(x)

class DecisionTreeRegressor(SkLearnModel):
    """
    Decision tree for regression

    Wrapper of sklearn.tree.DecisionTreeRegressor
    """
    def __init__(self, **kwargs):
        self.model = SkDecTreeR(**kwargs)
        self.result_type = 'regression'
    
    def predict(self, x):
        p = super().predict(x)
        if len(p.shape) == 1:
            return p.reshape((len(p), 1))

        return p
    