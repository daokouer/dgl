import networkx as nx
from mx import mx_Graph
from glimpse import create_glimpse
import torch as T
import torch.nn as nn
import torch.functional as F

t = nx.balanced_tree(2, 2)
t_uni = nx.bfs_tree(t, 0)

class MessageModule(nn.Module):
    def forward(self, state):
        h, b, y = state
        return h

class UpdateModule(nn.Module):
    def __init__(self,
                 h_dims=128,
                 n_classes=10,
                 steps=5,
                 filters=[16, 32, 64, 128, 256],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 glimpse_type='gaussian',
                 glimpse_size=(15, 15),
                 ):
        self.glimpse = create_glimpse(glimpse_type, glimpse_size)
        self.cnn = build_cnn(
                filters=filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size,
                )
        self.net_b = nn.Linear(h_dims, self.glimpse.att_params)
        self.net_y = nn.Linear(h_dims, n_classes)

    def set_image(self, x):
        self.x = x

    def forward(self, node_state, message):
        h, b, y = node_state
        b_new = b + self.net_b(h)
        y_new = y + self.net_y(h)

        g = self.glimpse(self.x, b_new)
        h_new = h + self.cnn(g) + message

        return h_new, b_new, y_new

def update_local():
    pass

class DFSGlimpseClassifier(nn.Module):
    def __init__(self, *args, **kwarg):
        nn.Module.__init__(self)
        t = nx.balanced_tree(2, 2)
        t_uni = nx.bfs_tree(t, 0)

        self.message_module = MessageModule()
        self.update_module = UpdateModule(
                h_dims=kwarg['h_dims'],
                n_classes=kwarg['n_classes'],
                glimpse_size=kwarg['glimpse_size'],
                )

        self.G = mx_Graph.DiGraph(t)
        self.G.register_update_module(self.update_module)
        self.G.register_message_module(self.message_module) # default: just copy
        self.G.register_readout_module(nn.Linear(h_dim, num_classes))

        root = 0
        self.e_list = []
        dfs_walk(t_uni, root, self.e_list)

    def forward(self, x):
        self.update_module.set_image(x)
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

