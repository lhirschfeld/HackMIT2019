import os
# Set any environment variables to limit MKL threads
import numpy as np

import ensemble_factory as ef

def make_uniform_child_generator(n):
    def uniform_child_generator():
        while True:
            yield n
    return uniform_child_generator()

class Genetic:
    def __init__(self, x, y, eval, popsize, keep, initialize, crossover, mutator, num_child_nodes_generator):
        self.x = x
        self.y = y
        self.num_features = x.shape[1]
        self.num_outputs = y.shape[0]
        self.population = dict()
        for member in initialize():
            self.population.setdefault(self.eval(member), []).append(member)
        self.crossover = crossover
        self.mutator = mutator
        self.eval = eval
        self.num_child_nodes_generator = num_child_nodes_generator

        self.popsize = popsize
        self.keep = keep

    def run(self, iters):
        best_loss = 1e15
        for n in range(iters):
            parents = []
            new_pop = dict()
            for key in sorted(self.population):
                if len(parents) >= self.keep:
                    break
                to_add = self.population[key][:min(self.keep-len(parents), len(self.population[key]))]
                new_pop[key] = to_add
                parents.extend(to_add)
            # TODO investigate other sampling methods than random sample... perhaps weighted sample?
            for i in range(self.popsize - self.keep):
                new_child = self.crossover(random.sample(parents, next(self.num_child_nodes_generator)))
                new_pop.setdefault(self.eval(new_child), []).append([])
            self.population = new_pop
            if min(self.population) < best_loss:
                best_loss = min(self.population)
                print(n, "New best loss", best_loss)
        return self.population[min(self.population)]
