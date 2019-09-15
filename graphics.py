import ensemble_factory as ef
import networkx as nx
import matplotlib.pyplot as plt
from ensemble_methods import *

def build_nx_graph(ensemble):
    queue = [ensemble]

    G = nx.DiGraph()
    G.add_node(str(ensemble))
    G.add_node('output')
    G.add_edge(str(ensemble),'output')

    while len(queue) > 0:
        node = queue.pop(0)

        if not hasattr(node, 'sub_ensembles'):
            continue

        
        for sub_ensemble in node.sub_ensembles:
            G.add_node(str(sub_ensemble))
            G.add_edge(str(sub_ensemble), str(node))

            queue.append(sub_ensemble)
    
    return G

def visualize_ensemble(ensemble):
    G = build_nx_graph(ensemble)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)

if __name__ == "__main__":
    regr = ef.adaboost(ef.decision_tree_regressor(), ef.bag(0.1,ef.linear_regression()), ef.adaboost(ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor()))
    visualize_ensemble(regr)
    plt.show()