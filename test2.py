import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

import ensemble_factory as ef
import sys

mnist = datasets.load_digits()

trX, teX, trY, teY = train_test_split(mnist.data / 255.0, mnist.target.astype("int0"), test_size = 0.33)

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

regr = ef.decision_tree_classifier()
accuracy = measure_accuracy(regr)
print(accuracy)

regr2 = ef.adaboost(ef.logistic_regression(), ef.bag(0.4, ef.decision_tree_classifier(), ef.decision_tree_classifier(), ef.decision_tree_classifier(), ef.decision_tree_classifier(), ef.decision_tree_classifier()))
accuracy2 = measure_accuracy(regr2)
print(accuracy2)