from ensemble import Ensemble
from sklearn.linear_model import LinearRegression, LogisticRegression

class LogRegression(Ensemble):
    def __init__(self):
        self.model = LogisticRegression()
    
    def _fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict_proba(x)

class LinRegression(Ensemble):
    def __init__(self):
        self.model = LinearRegression()
    
    def _fit(self, x, y):
        self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)


