import torch as th
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable as Var

import networkx as nx
from mx import mx_Graph as mg

th.random.manual_seed(0)
g = mg(nx.path_graph(3))
g.set_repr(0, th.rand(2, 4), name='x')
#g.print_all()

net = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid())

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(MLP, self).__init__()
        fcs = []
        h1 = input_dim
        for h2 in layer_sizes:
            fcs.append(nn.Linear(h1, h2))
            h1 = h2
        fcs.append(nn.Linear(h1, output_dim))
        self._fcs = nn.ModuleList(fcs)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        for i in range(len(self._fcs)-1):
            x = F.relu(self._fcs[i](x))
        x = self._fcs[-1](x)
        return x

x = th.rand(2, 4)
x = Var(x, requires_grad=True)
y = net(x)
net.zero_grad()
target = th.rand(2)
y.backward(th.rand(2))

# for p in net.parameters():
#     print p
#     #print p
    #print p, p.requires_grad
#loss = F.nll_loss(y, Var(th.ones(2).long()))
'''
1. Try F a loss function, and gradient
2. Study how Sean defines GRU module etc.
3. Do a sequntial net
4. Do a recurrent LM model
5. Define an attention graph
6. See how to build a graph with parameters
'''
