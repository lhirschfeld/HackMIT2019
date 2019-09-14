import abc
from random import random
import numpy as np
from sklearn.linear_model import LogisticRegression

class Ensemble:
    """
    Base class for ensembling.
    Instances of this class are capable of training and prediction.
    """    
    def fit(self, x, y):
        """
        Trains the ensemble using supplied data.
        """
        self._fit(x, y)

        # y_hat = self.predict(x)

        # self.variance = self._variance(y, y_hat)
        # self.bias = self._bias(y, y_hat)
        # self.mse = self._mse(y, y_hat)

        return self
    
    @abc.abstractmethod
    def _fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        """
        Predicts the labels of supplied data.
        """
        pass

    def _variance(self, y, y_hat):
        return [(a)]
        pass

    def _bias(self):
        pass

    def _mse(self):
        pass
    