import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import ensemble_factory as ef
from genetic import *

def main():
    is_classifier = False

    boston_data, boston_target = load_boston(return_X_y=True)
    boston_target = boston_target.reshape(-1, 1)

    data_train, data_test, target_train, target_test = train_test_split(boston_data, boston_target, test_size=0.2)

    data_train, target_train = unison_shuffled_copies(data_train, target_train)

    genetic = Genetic(
        make_default_eval(data_train, target_train, data_test, target_test), 
        popsize=30, 
        keep=10, 
        initialize=make_default_base_initialize(classifier=is_classifier), 
        crossover=make_joint_crossover([make_boost_crossover(is_classifier), make_simple_stack_crossover(is_classifier), bag_crossover], [1/3, 1/3, 1/3]),
        mutator=make_mutator(mutate_prob=0.05), 
        num_child_nodes_generator=make_uniform_child_generator(2), 
        run_name='boston_dataset'
    )
    ens = genetic.run(10)[0]

    print('Test loss:', ens._loss(data_test, target_test))
    print(ens.predict(data_test))
    print('Test accuracy:', np.sum(ens.predict(data_test) == target_test) / target_test.shape[1])

    ef.visualize_ensemble(ens)
    print(ens)
    plt.show()

if __name__ == '__main__':
    main()