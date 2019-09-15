import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

import ensemble_factory as ef
import sys

from genetic import Genetic, make_default_eval, default_base_initialize, make_joint_crossover, make_boost_crossover, make_simple_stack_crossover, bag_crossover, make_mutator, make_uniform_child_generator

mnist = datasets.load_digits()

trX, teX, trY, teY = train_test_split(mnist.data / 255.0, mnist.target.astype("int0"), test_size = 0.2)

oh_trY, oh_teY = [], []
for y in trY:
    one_hot = np.zeros(10)
    one_hot[y] = 1
    oh_trY.append(one_hot)

for y in teY:
    one_hot = np.zeros(10)
    one_hot[y] = 1
    oh_teY.append(one_hot)

oh_trY = np.array(oh_trY)
oh_teY = np.array(oh_teY)

trY = trY.reshape((-1, 1))
teY = teY.reshape((-1, 1))

def measure_accuracy(regr):
    regr.fit(trX, oh_trY)
    # print(trY.shape, teY.shape)
    # print(regr.predict(teX) == teY)
    # return np.sum(regr.predict(teX) == teY)/len(teY)
    # print(np.argmax(regr.predict(teX), axis=1))
    return sum(np.argmax(regr.predict(teX), axis=1) == teY.flatten())/len(teY)

is_classifier = True

genetic = Genetic(make_default_eval(trX[:int(len(trX)*0.75)], oh_trY[:int(len(trX)*0.75)], trX[int(len(trX)*0.75):], oh_trY[int(len(trX)*0.75):]), 30, 10, default_base_initialize, 
                make_joint_crossover([make_boost_crossover(is_classifier), make_simple_stack_crossover(is_classifier),bag_crossover], [1/3, 1/3, 1/3]),
                make_mutator(mutate_prob=0.05), make_uniform_child_generator(2), run_name='test' )
ens = genetic.run(5)[0]

print('test_loss', ens._loss(teX, oh_teY))
print('test_accuracy', sum(np.argmax(ens.predict(teX), axis=1) == teY.flatten())/len(teY))

ef.visualize_ensemble(ens)
print(ens)
plt.show()