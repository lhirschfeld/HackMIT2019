import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

import ensemble_factory as ef
import sys

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X = np.concatenate((diabetes_X,diabetes_X**2), axis=1)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

diabetes_y_train = diabetes_y_train.reshape((len(diabetes_y_train),1))
diabetes_y_test = diabetes_y_test.reshape((len(diabetes_y_test),1))


regr = ef.adaboost(ef.decision_tree_regressor(), ef.bag(0.1,ef.linear_regression()), ef.adaboost(ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor()))
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variances:', regr.variance, lin1.variance, lin2.variance)

# Plot outputs
sorted_vals = sorted([(diabetes_X_test[:,0][i], diabetes_y_pred[i]) for i in range(len(diabetes_y_pred))])
plt.scatter(diabetes_X_test[:,0], diabetes_y_test,  color='black')
plt.plot([val[0] for val in sorted_vals], [val[1] for val in sorted_vals], color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

ef.visualize_ensemble(regr)
plt.show()