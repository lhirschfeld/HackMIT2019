import ensemble_factory as ef
import networkx as nx
import matplotlib.pyplot as plt
from ensemble_methods import *
import pickle
import sys
from sklearn.model_selection  import train_test_split
import numpy as np
import random
from sklearn import datasets
import re
import matplotlib.colors as mcolors
import matplotlib.cm as cmx

cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}

cmap = mcolors.LinearSegmentedColormap(
'my_colormap', cdict, 100)


def build_nx_graph(ensemble, x, y):
    queue = [ensemble]

    node_losses = []

    def add_node(node):
        G.add_node(str(node))
        node_losses.append(-1*node._accuracy(x,y))

    G = nx.DiGraph()
    add_node(ensemble)

    while len(queue) > 0:
        node = queue.pop(0)

        if not hasattr(node, 'sub_ensembles'):
            continue

        
        for sub_ensemble in node.sub_ensembles:
            add_node(sub_ensemble)
            G.add_edge(str(sub_ensemble), str(node))

            queue.append(sub_ensemble)
    
    return G, node_losses

def visualize_ensemble(ensemble, x, y):
    G, losses = build_nx_graph(ensemble, x, y)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    labels = {str(node): ''.join(re.sub( r"([A-Z])", r" \1", str(node).split(' ')[0]).split()[:2]) for node in G.nodes}

    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)
    cNorm  = mcolors.Normalize(vmin=min(losses), vmax=max(losses))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    for accuracy in np.linspace(min(losses), max(losses),8):
        ax.plot([0],[0],color=scalarMap.to_rgba(accuracy),label=str(round(-1*accuracy,2))+'% acc')
    
    nx.draw(G, pos, with_labels=False, arrows=True, node_color=losses, node_size=800, cmap=cmap)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=5)
    
    plt.axis('off')
    f.set_facecolor('w')
    plt.legend(loc='lower right')
    f.tight_layout()




if __name__ == "__main__":
    # ens = ef.adaboost(ef.decision_tree_regressor(), ef.bag(0.1,ef.linear_regression()), ef.adaboost(ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor(),ef.linear_regression(),ef.decision_tree_regressor()))
    with open('mnist_small_ens.pickle', mode='rb') as f:
        ens = pickle.load(f)

    mnist = datasets.load_digits()

    trX, teX, trY, teY = train_test_split(mnist.data / 255.0, mnist.target.astype("int0"), test_size = 0.2)

    oh_teY = []
    for y in teY:
        one_hot = np.zeros(10)
        one_hot[y] = 1
        oh_teY.append(one_hot)

    oh_teY = np.array(oh_teY)

    visualize_ensemble(ens, teX, oh_teY)
    plt.show()