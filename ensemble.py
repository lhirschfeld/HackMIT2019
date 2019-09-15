import abc
from random import random
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

class Ensemble:
    """
    Base class for ensembling.
    Instances of this class are capable of training and prediction.
    """
    def __init__(self):
        self.depth = 1  # override this for ensemble methods

    def fit(self, x, y, **kwargs):
        """
        Trains the ensemble using supplied data.

        :param x (np.ndarray): the training data, with rows being individual observations
            Has shape (num_samples, num_cols)
        :param y (np.ndarray): the target for the training data, in the same order as x
            Has shape (num_samples, 1) 
        :return (Ensemble): the model
        """
        self._fit(x, y, **kwargs)

        self.train_loss = self._loss(x, y)

        return self
    
    @abc.abstractmethod
    def _fit(self, x, y, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, x):
        """
        Predicts the labels of supplied data.

        :param x (np.ndarray): the training data, with rows being individual observations
            Has shape (num_samples, num_cols)
        :return (np.ndarray): the predicted target values from the training data
            Has shape (num_samples, 1)
        """
        pass
    
    @abc.abstractmethod
    def copy(self):
        """
        Creates a copy of the existing ensemble.
        """
        pass

    def _loss(self, x, y):
        """
        Calculates and returns the training loss of the model

        Uses cross-entropy loss for classification models,
        and mean-squared error for regression models

        Assumes that the model has already called fit() to train,
        so predict() will give predictions based on the training data
        """
        preds = self.predict(x)
        if self.result_type == 'classification':
            return log_loss(y, preds)
        else:
            return mean_squared_error(y, preds)
    
    def __repr__(self):
        if not hasattr(self, 'id'):
            self.id = int(random()*(2**10))

        return self.__class__.__name__ + ' ' + str(self.id)