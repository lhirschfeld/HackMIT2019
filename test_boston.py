"""
Trying an arbitrary model on Boston dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import ensemble_factory as fact

boston_data, boston_target = load_boston(return_X_y=True)
boston_target = boston_target.reshape(-1, 1)

data_train, data_test, target_train, target_test = train_test_split(boston_data, boston_target, test_size=0.2)

forest = fact.bag(0.99, *[fact.decision_tree_regressor(max_depth=5) for _ in range(100)])
boosted_regressors = [fact.gradient_boost(*[fact.mlp_regressor(hidden_layer_sizes=(10,)) for _ in range(3)]) for _ in range(10)]
stack_boosted_regressors = fact.stack(fact.linear_regression(), *boosted_regressors)
model = fact.bag(0.99, forest, stack_boosted_regressors)

model.fit(data_train, target_train)

predictions = model.predict(data_test)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(target_test, predictions))
# Explained variance score: 1 is perfect prediction
#print('Variances:', regr.variance, lin1.variance, lin2.variance)

# Plot outputs
sorted_vals = sorted([(data_test[:,0][i], predictions[i]) for i in range(len(predictions))])
plt.scatter(data_test[:,0], target_test,  color='black')
plt.plot([val[0] for val in sorted_vals], [val[1] for val in sorted_vals], color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

fact.visualize_ensemble(model)
plt.show()
