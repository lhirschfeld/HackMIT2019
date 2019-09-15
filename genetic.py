import os
# Set any environment variables to limit MKL threads
import numpy as np
import pickle as pkl
import random
import matplotlib.pyplot as plt
import ensemble_factory as ef
from sklearn import datasets
import multiprocessing


def make_uniform_child_generator(n):
    def uniform_child_generator():
        while True:
            yield n
    return uniform_child_generator()

def make_boost_crossover(classifier=True):
    return ef.adaboost if classifier else ef.gradient_boost

def bag_crossover(*parents):
    return ef.bag(1.5/len(parents), *parents) # TODO tune the prob 1.5/len(parents)

def make_simple_stack_crossover(classifier=True):
    def stack_crossover(*parents):
        return ef.stack(ef.logistic_regression(), *parents) if classifier else ef.stack(ef.linear_regression(), *parents)
    return stack_crossover

def make_joint_crossover(crossovers, weights=None):
    if weights is None:
        weights = [1/len(crossovers)]*len(crossovers)
    def ret_crossover(parents):
        point = random.random()
        for i, w in enumerate(weights):
            if point < 0:
                return crossovers[i-1](*parents)
            point -= w
        return crossovers[-1](*parents)
    return ret_crossover
            
def make_default_eval(trainx, trainy, valx, valy):
    def default_eval(ensemble):
        ensemble.fit(trainx, trainy)
        return ensemble._loss(valx, valy)
    return default_eval

def make_mutator(mutate_prob=0.05, classifier=True):
    mutate_with = ef.BASE_CLASSIFIERS if classifier else ef.BASE_REGRESSORS
    def mutate(top):
        layer = [top]
        while len(layer) != 0:
            new_layer = []
            for ensemble in layer:
                if hasattr(ensemble, 'sub_ensembles'):
                    new_layer.extend(ensemble.sub_ensembles)
                    if random.random() < mutate_prob:
                        # print("mutating")
                        ensemble.sub_ensembles.append(random.choice(ef.BASE_CLASSIFIERS)())
            layer = new_layer
    return mutate

def make_default_base_initialize(classifier=True):
    def default_base_initialize():
        pop = []
        for method in (ef.BASE_CLASSIFIERS if classifier else ef.BASE_REGRESSORS):
            pop.append(method())
        return pop
    return default_base_initialize

class Genetic:
    def __init__(self, evaluate, popsize, keep, initialize, crossover, mutator, num_child_nodes_generator, is_classifier=True, run_name='default'):
        #self.num_features = x.shape[1]
        #self.num_outputs = y.shape[0]
        self.is_classifier=is_classifier
        self.population = dict()
        self.evaluate = evaluate
        for member in initialize():
            print(self.evaluate(member))
            self.population.setdefault(self.evaluate(member), []).append(member)
        self.crossover = crossover
        self.mutator = mutator
        self.num_child_nodes_generator = num_child_nodes_generator

        self.popsize = popsize
        self.keep = keep
        self.run_name = run_name

    def run(self, iters, add_simple=False):
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
                pass
            # TODO investigate other sampling methods than random sample... perhaps weighted sample?
            children = []
            if add_simple:
                for method in (ef.BASE_CLASSIFIERS if self.is_classifier else ef.BASE_REGRESSORS):
                    parents.append(method())
            for i in range(self.popsize - self.keep):
                group = random.sample(parents, next(self.num_child_nodes_generator))
                new_child = self.crossover([g.copy() for g in group])
                self.mutator(new_child)
                try:
                    #children.append(new_child)
                    new_pop.setdefault(self.evaluate(new_child), []).append(new_child)
                except:
                    print("Error skipping...")
                    i -= 1
                    continue
            
            # print(children)
            # pool = multiprocessing.Pool(multiprocessing.cpu_count())
            # perfs = pool.map(self.evaluate, children)

            # for perf, child in zip(perfs, children):
            #     new_pop.setdefault(perf, []).append(new_child)

            self.population = new_pop
            #print(n, self.population.keys())
            if min(self.population.keys()) < best_loss:
                best_loss = min(self.population)
                print(n, "New best loss", best_loss)
            if n % 10 == 0:
                pkl.dump(self.population, open(self.run_name+'-temp.pkl', 'wb'))
                os.rename(self.run_name+'-temp.pkl', self.run_name+'.pkl')
            
            if best_loss < 1e-10:
                print("Early stopping", n)
                break
        return self.population[min(self.population)]

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == "__main__":
    is_classifier = True
    seed = random.randint(0, 1000)
    random.seed(seed)
    print("random seed", seed)

    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = []
    for y in iris.target:
        one_hot = np.zeros(3)
        one_hot[y] = 1
        iris_y.append(one_hot)
    iris_y = np.array(iris_y)

    iris_x, iris_y = unison_shuffled_copies(iris_x, iris_y)

    genetic = Genetic(make_default_eval(iris_x[:80], iris_y[:80], iris_x[80:120], iris_y[80:120]), 30, 5, make_default_base_initialize(classifier=is_classifier), 
                    make_joint_crossover([make_boost_crossover(is_classifier), make_simple_stack_crossover(is_classifier),bag_crossover], [1/3, 1/3, 1/3]),
                    make_mutator(mutate_prob=0.05, classifier=is_classifier), make_uniform_child_generator(2), is_classifier=True, run_name='test' )
    ens = genetic.run(10, add_simple=True)[0]

    print('test_loss',ens._loss(iris_x[120:],iris_y[120:]))
    print('test_accuracy', sum(np.argmax(ens.predict(iris_x[120:]), axis=1) == np.argmax(iris_y[120:],axis=1))/len(iris_y[120:]))

    ef.visualize_ensemble(ens)
    print(ens)
    plt.show()