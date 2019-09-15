import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import numpy as np
import random
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.examples.tutorials.mnist import input_data

import ensemble_factory as ef
import sys

from genetic import Genetic, make_default_eval, make_default_base_initialize, make_joint_crossover, make_boost_crossover, make_simple_stack_crossover, bag_crossover, make_mutator, make_uniform_child_generator, make_random_child_generator

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

def downsample(images):
    downsampled_images = []
    for img_old in images:
        new_img = np.zeros((14, 14))
        # print(img[0].size)
        img = img_old.reshape((28, 28))
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                new_img[i//2, j//2] = (img[i, j] + img[i, j + 1] + img[i + 1, j] + img[i + 1, j + 1])/4
        downsampled_images.append(new_img)
    return downsampled_images

downsampled_train = downsample(mnist.train.images)
downsampled_test = downsample(mnist.test.images)

trX = np.vstack([img.reshape(-1,) for img in downsampled_train])
trY = mnist.train.labels

trX = trX[:int(len(trX)/10)]
trY = trY[:int(len(trY)/10)]

teX = np.vstack([img.reshape(-1,) for img in downsampled_test])
teY = mnist.test.labels

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

seed = 487#random.randint(0, 1000)
random.seed(seed)
print("random seed", seed)
is_classifier = True

genetic = Genetic(make_default_eval(trX[:int(len(trX)*0.75)], oh_trY[:int(len(trX)*0.75)], trX[int(len(trX)*0.75):], oh_trY[int(len(trX)*0.75):]), 60, 10, make_default_base_initialize(classifier=is_classifier), 
                make_joint_crossover([make_boost_crossover(is_classifier), make_simple_stack_crossover(is_classifier),bag_crossover], [1/3, 1/3, 1/3]),
                make_mutator(mutate_prob=0.05, classifier=is_classifier), make_random_child_generator([2,3,4], [1/3, 1/3, 1/3]), run_name='mnist-medium' )
ens = genetic.run(0, add_simple=True)[0]

trY = trY.reshape((-1, ))
teY = teY.reshape((-1, ))

# Baseline: random forest
rf = RandomForestClassifier(n_estimators=10)
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

ef.visualize_ensemble(ens, teX, oh_teY)
print(ens)
plt.show()