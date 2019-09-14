import os
# Set any environment variables to limit MKL threads
import numpy as np
import pickle as pkl
import random

import ensemble_factory as ef

def make_uniform_child_generator(n):
    def uniform_child_generator():
        while True:
            yield n
    return uniform_child_generator()

def make_boost_crossover(classifier=True):
    return ef.adaboost if classifier else ef.gradient_boost

def bag_crossover(parents):
    return ef.bag(1.5/len(parents), *parents) # TODO tune the prob 1.5/len(parents)

def make_simple_stack_crossover(classifier=True):
    def stack_crossover(parents):
        return ef.stack(ef.logistic_regression, *parents) if classifier else ef.stack(ef.linear_regression, *parents)

def make_joint_crossover(crossovers, weights=None):
    if weights is None:
        weights = [1/len(crossovers)]*len(crossovers)
    def ret_crossover(parents):
        point = random.random()
        for i, w in enumerate(weights):
            if point < 0:
                return crossovers[i-1](parents)
            point -= w
        return crossovers[-1](parents)
    return ret_crossover
            
def make_default_eval(x, y):
    def default_eval(ensemble):
        ensemble.fit(x, y)
        return ensemble._loss
    return default_eval

def make_mutator(mutate_prob=0.05):
    def mutate(top):
        layer = [top]
        while len(layer) != 0:
            new_layer = []
            for ensemble in layer:
                if hasattr(ensemble, 'sub_ensembles'):
                    new_layer.extend(ensemble.sub_ensembles)
                    if random.random() < mutate_prob:
                        ensemble.sub_ensembles.append(random.choice(ef.BASE_CLASSIFIERS)())
            layer = new_layer
    return mutate

def default_base_initialize():
    pop = []
    for method in ef.BASE_CLASSIFIERS:
        pop.append(method())
    return pop

class Genetic:
    def __init__(self, eval, popsize, keep, initialize, crossover, mutator, num_child_nodes_generator, run_name='default'):
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
            if len(parents) < self.keep:

            # TODO investigate other sampling methods than random sample... perhaps weighted sample?
            for i in range(self.popsize - self.keep):
                new_child = self.crossover(random.sample(parents, next(self.num_child_nodes_generator)))
                self.mutator(new_child)
                new_pop.setdefault(self.eval(new_child), []).append([])
            self.population = new_pop
            if min(self.population) < best_loss:
                best_loss = min(self.population)
                print(n, "New best loss", best_loss)
            if n % 10 == 0:
                pkl.dump(self.population, open(run_name+'-temp.pkl', 'wb'))
                os.rename(run_name+'-temp.pkl', run_name+'.pkl')
        return self.population[min(self.population)]

if __name__ == 'main':
    is_classifier = True
    genetic = Genetic(make_default_eval(X, Y), 100, 20, default_base_initialize, 
                    make_joint_crossover([make_boost_crossover(is_classifier,) make_simple_stack_crossover(is_classifier),bag_crossover], [1/3, 1/3, 1/3]),
                    make_mutator(mutate_prob=0.05), make_uniform_child_generator(2), run_name='test' )
    genetic.run(5)
