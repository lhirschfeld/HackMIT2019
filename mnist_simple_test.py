import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import numpy as np
import random
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import ensemble_factory as ef
import sys

from genetic import Genetic, make_default_eval, make_default_base_initialize, make_joint_crossover, make_boost_crossover, make_simple_stack_crossover, bag_crossover, make_mutator, make_uniform_child_generator, make_random_child_generator

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

seed = random.randint(0, 1000)
random.seed(seed)
print("random seed", seed)
is_classifier = True

genetic = Genetic(make_default_eval(trX[:int(len(trX)*0.75)], oh_trY[:int(len(trX)*0.75)], trX[int(len(trX)*0.75):], oh_trY[int(len(trX)*0.75):]), 60, 10, make_default_base_initialize(classifier=is_classifier), 
                make_joint_crossover([make_boost_crossover(is_classifier), make_simple_stack_crossover(is_classifier),bag_crossover], [1/3, 1/3, 1/3]),
                make_mutator(mutate_prob=0.05, classifier=is_classifier), make_random_child_generator([2,3,4], [1/3, 1/3, 1/3]), run_name='test' )
ens = genetic.run(5, add_simple=True)[0]

# Baseline: random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(trX, trY)
rf_acc = sum(teY == rf.predict(teX))/len(teY)

# Baseline: logistic regression
logreg = LogisticRegression()
logreg.fit(trX, trY)
logreg_acc = sum(teY == logreg.predict(teX)) / len(teY)

print('test_loss', ens._loss(teX, oh_teY))
print('test_accuracy', ens._accuracy(teX, oh_teY))
print('rf_accuracy', rf_acc)
print('logreg_accuracy', logreg_acc)

ef.visualize_ensemble(ens)
print(ens)
plt.show()