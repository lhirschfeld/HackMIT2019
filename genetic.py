import os
# Set any environment variables to limit MKL threads
import numpy as np

import ensemble_factory as ef

class Genetic:
    def __init__(self, x, y, popsize, keep, initialize, crossover, mutator):
        self.x = x
        self.y = y
        self.num_features = x.shape[1]
        self.num_outputs = y.shape[0]
        self.population = initialize()
        self.population.sort(key=lambda)
        self.crossover = crossover
        self.mutator = mutator

        self.keep = keep

    def eval(self):
        for ensemble in self.population:
            ensemble.fit
    

    def run(self, iters):
        for n in range(iters):
            len(population)
