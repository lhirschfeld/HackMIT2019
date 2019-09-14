import ensemble_factory as ef
import networkx as nx

regr = ef.adaboost(ef.decision_tree_regressor(), ef.bag(0.1,ef.linear_regression()), ef.adaboost(ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor()))

def build_nx_graph(ensemble):
    queue = []
    ensemble.sub_ensembles