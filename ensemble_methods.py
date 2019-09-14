from ensemble import Ensemble
from base_models import *
import numpy as np

class Bag(Ensemble):
    def __init__(self, sample_prob, *sub_ensembles):
        self.sample_prob = sample_prob
        self.sub_ensembles = sub_ensembles
    
    def fit(self, x, y):
        def subsample_with_replacement():
            x_sample = []
            y_sample = []
            for x_i, y_i in zip(x, y):
                if random() < self.sample_prob:
                    x_sample.append(x_i)
                    y_sample.append(y_i)
            
            return np.array(x_sample), np.array(y_sample)
        
        for ensemble in self.sub_ensembles:
            x_s, y_s = subsample_with_replacement()
            ensemble.fit(x_s, y_s)
            
    def predict(self, x):
        preds = np.array([ensemble.predict(x) for ensemble in self.sub_ensembles])
        return np.mean(preds, axis=0)

class Boost(Ensemble):
    def __init__(self, sub_ensemble, booster):
        self.sub_ensemble = sub_ensemble
        self.booster = booster
    
    def _fit(self, x, y):
        pass


class Stack(Ensemble):
    def __init__(self, stack_model, *sub_ensembles):
        self.sub_ensembles = sub_ensembles
        self.model = stack_model
    
    def _fit(self, x, y):
        preds = [ensemble.fit(x, y).predict(x) for ensemble in self.sub_ensembles]
        augmented_x = np.array([x[i] + [p[i] for p in preds] for i in range(len(x))])
        
        self.model.fit(augmented_x, y)
    
    def predict(self, x):
        preds = [ensemble.predict(x) for ensemble in self.sub_ensembles]
        augmented_x = np.array([x[i] + [p[i] for p in preds] for i in range(len(x))])

        return self.model.predict(augmented_x)
        



