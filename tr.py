import networkx as nx
from mx import mx_Graph
import torch as T
import torch.nn as nn
import torch.functional as F

t = nx.balanced_tree(2, 2)
t_uni = nx.bfs_tree(t, 0)

def update_local():
    pass

class DSFGlimpseClassifier(nn.Module):
    def __init__(self, *args, **kwarg):
        nn.Module.__init__(self)
        t = nx.balanced_tree(2, 2)
        t_uni = nx.bfs_tree(t, 0)

        self.G = mx_Graph.DiGraph(t)
        self.G.register_update_module()
        self.G.register_message_module() # default: just copy
        self.G.register_readout_module(nn.Linear(h_dim, num_classes))

        root = 0
        self.e_list = []
        dfs_walk(t_uni, root, self.e_list)

    def forward(self, x):
        pred = []
        y = self.classifier(self.G.get_repr(root))
        pred.append(y)
        for u, v in self.e_list:
            self.G.update_by_edge((v, u))
            for i in T_MAX_RECUR:
                self.G.update_local(u)
            y = self.classifier(self.G.get_repr(u))
            pred.append(y)
        return pred

model = mx_Graph(t)
model.register_update_func()

dfs_edge_list = []
def dfs_walk(tree, curr, l):
    if len(tree.succ[curr]) == 0:
        return
    else:
        for n in tree.succ[curr]:
            l.append((curr, n))
            dfs_walk(tree, n, l)
            l.append((n, curr))

dfs_walk(t_uni, 0, dfs_edge_list)
'''
[(0, 1),
 (1, 3),
 (3, 1),
 (1, 4),
 (4, 1),
 (1, 0),
 (0, 2),
 (2, 5),
 (5, 2),
 (2, 6),
 (6, 2),
 (2, 0)]
'''
for u, v in dfs_edge_list:
    model.update_by_edge((v, u))

