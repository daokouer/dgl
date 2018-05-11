import networkx as nx

'''
a simple feed forward network
'''
L = 10

from networkx.classes.graph import Graph

class mx_Graph(Graph):
    def __init__(self):
        super(mx_Graph).__init__()


net = mx_nx.path_graph(L)

mx_g.set_update_func(ReLU)
mx_g.set_msg_func(MLP)

config = {}

for i, n in enumerate(net):
    net.add_node(i, func=mx_g, config[i])

for e in range(num_epoches):
    for x in batches:
        y = net.forward(x)
        loss = loss_fn(y, y_label)
        loss.backward()
        net.update_param()

'''
with skip connection
'''

'''
a recurrent network for LM
'''

'''
a recurrent network for LM
'''
import networkx as nx
import mx

net = mx.mx_Graph(nx.path_graph(t_max))
net.set_update_func(func=F.GRU)

for i, w in enumerate(sent):
    net.set_repr(i, w)
net.update_from(0, mode=all)


'''
a recurrent network for NTM with attn
'''

'''
a fwd network composed of some building blocks
'''


