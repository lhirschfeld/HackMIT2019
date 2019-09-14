import abc

class Ensemble:
    """
    Base class for ensembling.
    Instances of this class are capable of training and prediction.
    """

    def __init__(self):
        return

    @staticmethod
    def bag(sub_ensembles, splits):
        return Bag(sub_ensembles, splits)

    @staticmethod
    def boost(sub_ensemble):
        return Boost(sub_ensemble)

    @staticmethod
    def stack(sub_ensembles):
        return Stack(sub_ensembles)
    
    @abc.abstractmethod
    def fit(self, data):
        """
        Trains the ensemble using supplied data.
        """
        pass

    @abc.abstractmethod
    def predict(self, data):
        """
        Predicts the labels of supplied data.
        """
        pass

class Bag(Ensemble):
    def __init__(self, sub_ensembles, splits):
        self.sub_ensembles = sub_ensembles
        self.splits = splits


class Boost(Ensemble):
    def __init__(self, sub_ensemble):
        self.sub_ensemble = sub_ensemble


class Stack(Ensemble):
    def __init__(self, sub_ensembles):
        self.sub_ensembles = sub_ensembles