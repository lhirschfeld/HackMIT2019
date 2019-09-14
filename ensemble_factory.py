import numpy as np
import ensemble_methods
import base_models
import graphics

def logistic_regression(**kwargs):
    """
    Returns a logistic regression classifier
    """
    return base_models.LogRegression(**kwargs)

def linear_regression(**kwargs):
    """
    Returns a linear regression solver
    """
    return base_models.LinRegression(**kwargs)

def lasso_regressor(**kwargs):
    """
    Returns a lasso regressor
    """
    return base_models.LassoRegressor(**kwargs)

def ridge_regressor(**kwargs):
    """
    Returns a ridge regressor
    """
    return base_models.RidgeRegressor(**kwargs)

def svm_classifier(**kwargs):
    """
    Returns a SVM classifier
    """
    return base_models.LinearSVMClassifier(**kwargs)

def svm_regressor(**kwargs):
    """
    Returns a SVM regressor
    """
    return base_models.LinearSVMRegressor(**kwargs)

def mlp_classifier(**kwargs):
    """
    Returns a multi-layer perceptron classifier
    """
    return base_models.MLPClassifier(**kwargs)

def mlp_regressor(**kwargs):
    """
    Returns a multi-layer perceptron classifier
    """
    return base_models.MLPRegressor(**kwargs)

def decision_tree_classifier(**kwargs):
    """
    Returns a decision tree classifier
    """
    return base_models.DecisionTreeClassifier(**kwargs)

def decision_tree_regressor(**kwargs):
    """
    Returns a decision tree regressor
    """
    return base_models.DecisionTreeRegressor(**kwargs)

BASE_CLASSIFIERS = [logistic_regression, svm_classifier, decision_tree_classifier] # TODO put MLP back in at some point

def bag(sample_prob, *sub_ensembles):
    """
    Returns a bagging ensemble model using the given 
    sampling probability and submodels to bag

    :param sample_prob (float): the probability that an observation is sampled
        for the training data of a submodel
    :param *sub_ensembles (*Ensemble): the models of which the bagging uses the outputs
    :return (Bag(Ensemble)): the bagging ensemble model
    """
    return ensemble_methods.Bag(sample_prob, sub_ensembles)

def gradient_boost(*sub_ensembles):
    """
    Returns an ensemble model that chains the specified models using gradient boosting

    :param *sub_ensembles (*Ensemble): the models that are chained with gradient boosting
        The first model is the earliest in the ensemble graph, 
        with the second model taking the output of the first, etc.
    :return (GradientBoost(Ensemble)): the gradient boosting ensemble model
    """
    return ensemble_methods.GradientBoost(sub_ensembles)

def adaboost(*sub_ensembles):
    """
    Returns an ensemble model that chains the specified models using gradient boosting

    :param *sub_ensembles (*Ensemble): the models that are chained with gradient boosting
        The first model is the earliest in the ensemble graph, 
        with the second model taking the output of the first, etc.
    :return (AdaBoost(Ensemble)): the AdaBoost ensemble model
    """
    return ensemble_methods.AdaBoost(sub_ensembles)

def stack(stack_model, *sub_ensembles):
    """
    Returns a bagging ensemble model using the given stack model and submodels to stack
    
    :param stack_model (Ensemble): the model used to combine the results of the submodels
    :param *sub_ensembles (*Ensemble): the models of which the bagging uses the outputs
    :return (Stack(Ensemble)): the stack ensemble model
    """
    return ensemble_methods.Stack(stack_model, sub_ensembles)

def build_nx_graph(ensemble):
    """
    Returns a NetworkX graph representation of the ensemble
    
    :param stack_model (Ensemble): the ensemble to visualize
    :return (NetworkX.DiGraph): the graph
    """

    return graphics.build_nx_graph(ensemble)

def visualize_ensemble(ensemble):
    """
    Plots to matplotlib the NetworkX graph of the ensemble.
    
    :param stack_model (Ensemble): the ensemble to visualize
    """

    graphics.visualize_ensemble(ensemble)



    