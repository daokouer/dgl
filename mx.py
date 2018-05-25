import networkx as nx
from networkx.classes.digraph import DiGraph


# TODO: loss functions and training

'''
Defult modules: this is Pytorch specific
    - MessageModule: copy
    - UpdateModule: vanilla RNN
    - ReadoutModule: bag of words
    - ReductionModule: bag of words
'''
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DefaultMessageModule(nn.Module):
    """
    Default message module:
        - copy
    """
    def __init__(self, *args, **kwargs):
        super(DefaultMessageModule, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x

class DefaultUpdateModule(nn.Module):
    """
    Default update module:
        - a vanilla GRU with ReLU
    """
    def __init__(self, *args, **kwargs):
        super(DefaultUpdateModule, self).__init__(*args, **kwargs)
        h_dims = self.h_dims = kwargs.get('h_dims', 128)
        self.fc = nn.Linear(2 * h_dims, h_dims)

    def forward(self, x, m):
        #FIXME: ugly... in case x is not initialized yet
        if x == None:
            x = th.zeros_like(m)
        _in = th.cat((x, m), 1)
        out = F.relu(self.fc(_in))
        return out

class DefaultReadoutModule(nn.Module):
    """
    Default readout:
        - bag of words
    """
    def __init__(self, *args, **kwargs):
        super(DefaultReadoutModule, self).__init__(*args, **kwargs)

    def forward(self, x_s):
        out = th.stack(x_s)
        out = th.sum(out, dim=0)
        return out

class mx_Graph(DiGraph):
    '''
    Functions:
        - m_func: per edge (u, v), default is u['state']
        - u_func: per node u, default is RNN(m, u['state'])
    '''
    def __init__(self, *args, **kargs):
        super(mx_Graph, self).__init__(*args, **kargs)
        self.m_func = DefaultMessageModule()
        self.u_func = DefaultUpdateModule()
        self.readout_func = DefaultReadoutModule()
        # FIXME: reduction func reuse bag of words
        self.reduction_func = DefaultReadoutModule()
        self.init_reprs()

    def init_reprs(self, h_init=None):
        for n in self.nodes:
            self.set_repr(n, h_init)

    def set_repr(self, u, h_u, name='state'):
        assert u in self.nodes
        kwarg = {name: h_u}
        self.add_node(u, **kwarg)

    def get_repr(self, u, name='state'):
        assert u in self.nodes
        return self.nodes[u][name]

    def _nodes_or_all(self, nodes='all'):
        return self.nodes() if nodes == 'all' else nodes

    def _edges_or_all(self, edges='all'):
        return self.edges() if edges == 'all' else edges

    def register_message_func(self, message_func, edges='all', batched=False):
        '''
        batched: whether to do a single batched computation instead of iterating
        message function: accepts source state tensor and edge tag tensor, and
        returns a message tensor
        '''
        if edges == 'all':
            self.m_func = message_func
        else:
            for e in self.edges:
                self.edges[e]['m_func'] = message_func

    def register_update_func(self, update_func, nodes='all', batched=False):
        '''
        batched: whether to do a single batched computation instead of iterating
        update function: accepts a node attribute dictionary (including state and tag),
        and a list of tuples (source node, target node, edge attribute dictionary)
        '''
        if nodes == 'all':
            self.u_func = update_func
        else:
            for n in nodes:
                self.node[n]['u_func'] = update_func

    def register_readout_func(self, readout_func):
        self.readout_func = readout_func

    def readout(self, nodes='all'):
        nodes_state = []
        nodes = self._nodes_or_all(nodes)
        for n in self.nodes:
            nodes_state.append(self.get_repr(n))
        return self.readout_func(nodes_state)

    def sendto(self, u, v):
        """Compute message on edge u->v
        Args:
            u: source node
            v: destination node
        """
        try:
            f_msg = self.edges[(u, v)]['m_func']
        except KeyError:
            f_msg = self.m_func
        m = f_msg(self.get_repr(u))
        self.edges[(u, v)]['msg'] = m

    def recvfrom(self, u, nodes):
        """Update u by nodes
        Args:
            u: node to be updated
            nodes: nodes with pre-computed messages to u
        """
        m = [self.edges[(v, u)]['msg'] for v in nodes]
        try:
            f_update = self.nodes[u]['u_func']
        except KeyError:
            f_update = self.u_func

        #FIXME: we have a problem here...
        #FIXME: we need a reduction function to deal with variable
        #FIXME: number of neighbors.
        msg_gathered = self.reduction_func(m)
        x_new = f_update(self.get_repr(u), msg_gathered)
        self.set_repr(u, x_new)

    def update_by_edge(self, e):
        u, v = e
        self.sendto(u, v)
        self.recvfrom(v, [u])

    def update_to(self, u):
        """Pull messages from 1-step away neighbors of u"""
        assert u in self.nodes

        for v in self.pred[u]:
            self.sendto(v, u)
        self.recvfrom(u, list(self.pred[u]))

    def update_from(self, u):
        """Update u's 1-step away neighbors"""
        assert u in self.nodes
        for v in self.succ[u]:
            self.update_to(v)

    def draw(self):
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(self, prog='dot')
        nx.draw(self, pos, with_labels=True)

    def set_reduction_func(self):
        def _default_reduction_func(x_s):
            out = th.stack(x_s)
            out = th.sum(out, dim=0)
            return out
        self._reduction_func = _default_reduction_func

    def set_gather_func(self, u=None):
        pass

    def print_all(self):
        for n in self.nodes:
            print(n, self.nodes[n])
        print()

if __name__ == '__main__':
    from torch.autograd import Variable as Var

    th.random.manual_seed(0)

    g_path = mx_Graph(nx.path_graph(2))
    g_path.set_repr(0, th.rand(2, 128))
    g_path.sendto(0, 1)
    g_path.recvfrom(1, [0])
    g_path.readout()

    '''
    # this makes a uni-edge tree
    tr = nx.bfs_tree(nx.balanced_tree(2, 3), 0)
    m_tr = mx_Graph(tr)
    m_tr.print_all()
    '''
    print("testing GRU update")
    g = mx_Graph(nx.path_graph(3))
    g.register_update_func(nn.GRUCell(4, 4))
    fwd_net = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    g.register_message_func(fwd_net)

    for n in g:
        g.set_repr(n, th.rand(2, 4))

    y_pre = g.readout()
    g.update_from(0)
    y_after = g.readout()

