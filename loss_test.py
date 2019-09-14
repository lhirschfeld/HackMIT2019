import numpy as np

from ensemble import Ensemble
import base_models as models
import ensemble_factory as factory

def main():
    x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    y = np.array([1, 2, 2])

    logreg = models.LogRegression()
    logreg.fit(x, y)
    print(logreg.predict(x))
    print(logreg._loss(x, y))
    print(logreg.train_loss)

if __name__ == '__main__':
    main()
