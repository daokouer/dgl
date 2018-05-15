import networkx as nx
from mx import mx_Graph

t = nx.balanced_tree(2, 2)
t_uni = nx.bfs_tree(t, 0)

def update_local():
    pass

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

