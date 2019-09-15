import numpy as np
from random import random

from ensemble import Ensemble
from base_models import *

class Bag(Ensemble):
    def __init__(self, sample_prob, sub_ensembles):
        self.sub_ensembles = list(sub_ensembles)
        assert len(sub_ensembles) > 0, "Must specify at least one submodel for %s" % (
            self.__class__.__name__)
        assert len(set([e.result_type for e in sub_ensembles])) == 1, "All submodel result_types must match"
        self.sample_prob = sample_prob
        self.result_type = self.sub_ensembles[0].result_type
        self.depth = max([getattr(en, 'depth', 1) for en in sub_ensembles])
    
    def _fit(self, x, y, **kwargs):
        weights = kwargs['sample_weight'] if 'sample_weight' in kwargs else np.ones(len(x))
        
        def subsample_with_replacement():
            x_sample = []
            y_sample = []
            w_sample = []

            for x_i, y_i, w_i in zip(x, y, weights):
                if random() < self.sample_prob:
                    x_sample.append(x_i)
                    y_sample.append(y_i)
                    w_sample.append(w_i)
            
            return np.array(x_sample), np.array(y_sample), np.array(w_sample)
        
        for ensemble in self.sub_ensembles:
            x_s, y_s, w_s = subsample_with_replacement()
            ensemble.fit(x_s, y_s, sample_weight=w_s)
            
    def predict(self, x):
        preds = np.array([ensemble.predict(x) for ensemble in self.sub_ensembles])
        return np.mean(preds, axis=0)
    
    def copy(self):
        return Bag(self.sample_prob, [ensemble.copy() for ensemble in self.sub_ensembles])

class GradientBoost(Ensemble):
    def __init__(self, sub_ensembles):
        self.sub_ensembles = list(sub_ensembles)

        # Implicit: First sub_ensemble is being boosted
        assert len(sub_ensembles) > 0, "Must specify at least one submodel for %s" % (
            self.__class__.__name__)
        assert len(set([e.result_type for e in sub_ensembles])) == 1, "All submodel result_types must match"
        self.result_type = self.sub_ensembles[0].result_type
        self.depth = max([getattr(en, 'depth', 1) for en in sub_ensembles])
    
    def _fit(self, x, y, **kwargs):
        residuals = y
        self._output_size = len(y[0])
        for sub_ensemble in self.sub_ensembles:
            residuals -= sub_ensemble.fit(x, residuals, **kwargs).predict(x)
    
    def predict(self, x):
        preds = np.zeros((len(x), self._output_size))
        for sub_ensemble in self.sub_ensembles:
            preds += sub_ensemble.predict(x)
        
        return preds
    
    def copy(self):
        return GradientBoost([ensemble.copy() for ensemble in self.sub_ensembles])

class AdaBoost(Ensemble):
    def __init__(self, sub_ensembles):
        self.sub_ensembles = list(sub_ensembles)

        assert len(sub_ensembles) > 0, "Must specify at least one submodel for %s" % (
            self.__class__.__name__)
        assert len(set([e.result_type for e in sub_ensembles])) == 1, "All submodel result_types must match"
        self.result_type = self.sub_ensembles[0].result_type
        self.depth = max([getattr(en, 'depth', 1) for en in sub_ensembles])
    
    def _fit(self, x, y, **kwargs):
        if 'sample_weight' in kwargs:
            weights = kwargs['sample_weight']
        else:
            weights = np.ones(len(y))

        for ensemble in self.sub_ensembles:
            preds = ensemble.fit(x, y, sample_weight=weights).predict(x)
            
            if self.result_type == 'classification':
                preds *= 0.98
                preds += 0.001
                weights = -1 * ((y == 1) * np.log(preds) + (y != 1) * np.log(1-preds))
            else:
                weights = (preds - y)**2
            
            weights = np.sum(weights,axis=-1)

    def predict(self, x):
        preds = np.array([ensemble.predict(x) for ensemble in self.sub_ensembles])
        return np.mean(preds, axis=0)

    def copy(self):
        return AdaBoost([ensemble.copy() for ensemble in self.sub_ensembles])

class Stack(Ensemble):
    def __init__(self, stack_model, sub_ensembles):
        self.sub_ensembles = list(sub_ensembles)

        assert len(sub_ensembles) > 0, "Must specify at least one submodel for %s" % (
            self.__class__.__name__)
        assert len(set([e.result_type for e in self.sub_ensembles] + [stack_model.result_type])) == 1, "All submodel result_types must match"
        self.model = stack_model
        self.result_type = self.sub_ensembles[0].result_type
        self.depth = max([getattr(en, 'depth', 1) for en in sub_ensembles]) + getattr(stack_model, 'depth', 1)
    
    def _fit(self, x, y, **kwargs):
        preds = [ensemble.fit(x, y, **kwargs).predict(x) for ensemble in self.sub_ensembles]
        new_features = np.array([np.array([p[i, :] for p in preds]).flatten() for i in range(len(x))])
        augmented_x = np.concatenate((x, new_features), axis=1)
        
        self.model.fit(augmented_x, y)
    
    def predict(self, x):
        preds = [ensemble.predict(x) for ensemble in self.sub_ensembles]
        new_features = np.array([np.array([p[i, :] for p in preds]).flatten() for i in range(len(x))])
        augmented_x = np.concatenate((x, new_features), axis=1)

        return self.model.predict(augmented_x)
    
    def copy(self):
        return Stack(self.stack_model, [ensemble.copy() for ensemble in self.sub_ensembles])
