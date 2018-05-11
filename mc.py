import numpy as np
import networkx as nx

class MyClass:
    def __init__(self, h=[]):
        self._h = h

    def get_h(self):
        return self._h

    def set_h(self, h):
        self._h[:] = h[:]

c_list = [MyClass(list(range(i))) \
          for i in range(10)]

'''
adding attributes to node
'''
h = np.arange(9).reshape((3, 3))
g = nx.path_graph(3)
for n in g:
    g.add_node(n, h=h[n, :])

gx = nx.path_graph(3)
for i, x in enumerate(h):
    gx.add_node(i, h=x)
'''
Goals:
    - a path graph
    - draw path with labels
'''
g.clear()
g = nx.path_graph(4)
label_dict = {i: i+1 for i in range(4)}
nx.draw_networkx(g, labels=label_dict)

'''
    - make a GRU
    - a path graph, each node is a hidden state
    - update hidden state as in language modelling
'''

