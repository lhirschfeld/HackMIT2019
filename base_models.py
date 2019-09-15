import numpy as np
from sklearn.linear_model import LinearRegression as SkLinReg, SGDClassifier as SkSGD, \
    Lasso as SkLasso, Ridge as SkRidge
from sklearn.svm import LinearSVC as SkSVC, LinearSVR as SkSVR
from sklearn.neural_network import MLPClassifier as SkMLPC, MLPRegressor as SkMLPR
from sklearn.tree import DecisionTreeClassifier as SkDecTreeC, DecisionTreeRegressor as SkDecTreeR

from ensemble import Ensemble


class Average(Ensemble):
    def __init__(self, result_type):
        super(Average, self).__init__()
        self.result_type = result_type
        self.average = None
    
    def _fit(self, x, y, **kwargs):
        self.average = np.mean(y, axis=0)

    def predict(self, x):
        return np.array([self.average for _ in x])

class SkLearnModel(Ensemble):
    def __init__(self):
        super(SkLearnModel, self).__init__()
    
    def _fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)

class LogRegression(SkLearnModel):
    """
    Logistic regression 

    Wrapper of sklearn.linear_model.LogisticRegression
    """
    def __init__(self, **kwargs):
        super(LogRegression, self).__init__()
        self.model = SkSGD(loss="log", max_iter=1e2, **kwargs)
        self.result_type = 'classification'
    
    def _fit(self, x, y, **kwargs):
        self.model.fit(x, np.argmax(y, axis=1), **kwargs)
    
    def predict(self, x):
        return self.model.predict_proba(x)
    
class LinRegression(SkLearnModel):
    """
    Linear regresssion

    Wrapper of sklearn.linear_model.LinearRegression
    """
    def __init__(self, **kwargs):
        super(LinRegression, self).__init__()
        self.model = SkLinReg(**kwargs)
        self.result_type = 'regression'

class LassoRegressor(SkLearnModel):
    """
    Linear model with L1 regularization

    Wrapper of sklearn.linear_model.Lasso
    """
    def __init__(self, **kwargs):
        super(LassoRegressor, self).__init__()
        self.model = SkLasso(**kwargs)
        self.result_type = 'regression'

class RidgeRegressor(SkLearnModel):
    """
    Linear model with L2 regularization

    Wrapper of sklearn.linear_model.Ridge
    """
    def __init__(self, **kwargs):
        super(RidgeRegressor, self).__init__()
        self.model = SkRidge(**kwargs)
        self.result_type = 'regression'

class LinearSVMClassifier(SkLearnModel):
    """
    Support vector machine for classification

    Wrapper of sklearn.svm.LinearSVC
    """
    def __init__(self, **kwargs):
        super(LinearSVMClassifier, self).__init__()
        self.model = SkSVC(**kwargs)
        self.result_type = 'classification'
    
    def _fit(self, x, y, **kwargs):
        self.y_size = len(y[0])
        self.model.fit(x, np.argmax(y, axis=1), **kwargs)
    
    def predict(self, x):
        classes = super().predict(x)
        oh_y = []
        for y in classes:
            one_hot = np.zeros(self.y_size)
            one_hot[y] = 1
            oh_y.append(one_hot)
        
        return np.array(oh_y)



    
class LinearSVMRegressor(SkLearnModel):
    """
    Support vector machine for regression

    Wrapper of sklearn.svm.LinearSVR
    """
    def __init__(self, **kwargs):
        super(LinearSVMRegressor, self).__init__()
        self.model = SkSVR(**kwargs)
        self.result_type = 'regression'
    
class MLPClassifier(SkLearnModel):
    """
    Multi-layer perceptron for classification

    Wrapper of sklearn.neural_network.MLPClassification
    """
    def __init__(self, **kwargs):
        super(MLPClassifier, self).__init__()
        self.model = SkMLPC(**kwargs)
        self.result_type = 'classification'
    
    def _fit(self, x, y, **kwargs):
        y = y.flatten()
        self.model.fit(x, y, **kwargs)
    
class MLPRegressor(SkLearnModel):
    """
    Multi-layer perceptron for regression

    Wrapper of sklearn.neural_network.MLPRegressor
    """
    def __init__(self, **kwargs):
        super(MLPRegressor, self).__init__()
        self.model = SkMLPR(**kwargs)
        self.result_type = 'regression'
    
    def _fit(self, x, y, **kwargs):
        y = y.flatten()
        self.model.fit(x, y, **kwargs)
    
class DecisionTreeClassifier(SkLearnModel):
    """
    Decision tree for classification

    Wrapper of sklearn.tree.DecisionTreeClassifier
    """
    def __init__(self, **kwargs):
        super(DecisionTreeClassifier, self).__init__()
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
        super(DecisionTreeRegressor, self).__init__()
        self.model = SkDecTreeR(**kwargs)
        self.result_type = 'regression'
    
    def predict(self, x):
        p = super().predict(x)
        if len(p.shape) == 1:
            return p.reshape((len(p), 1))

        return p
    