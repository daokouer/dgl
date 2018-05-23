import networkx as nx
from networkx.classes.digraph import DiGraph


# TODO: loss functions and training

class mx_Graph(DiGraph):
    '''
    Functions:
        - m_func: per edge (u, v), default is u['state']
        - u_func: per node u, default is RNN(m, u['state'])
    '''
    def __init__(self, *args, **kargs):
        super(mx_Graph, self).__init__(*args, **kargs)
        self.set_msg_func()
        self.set_gather_func()
        self.set_reduction_func()
        self.set_update_func()
        self.set_readout_func()
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
        ebunch = self._edges_or_all(edges)
        for e in ebunch:
            self.edges[e]['m_func'] = message_func

    def register_update_func(self, update_func, nodes='all', batched=False):
        '''
        batched: whether to do a single batched computation instead of iterating
        update function: accepts a node attribute dictionary (including state and tag),
        and a list of tuples (source node, target node, edge attribute dictionary)
        '''
        nodes = self._nodes_or_all(nodes)
        for n in nodes:
            self.node[n]['u_func'] = update_func

    def register_readout_func(self, readout_func):
        self.readout_func = readout_func

    def readout(self):
        nodes_state = []
        for n in self.nodes:
            nodes_state.append(self.get_repr(n))
        return self.readout_func(nodes_state)

    def sendto(self, u, v):
        f_msg = self.edges[(u, v)]['m_func']
        m = f_msg(self.get_repr(u))
        self.edges[(u, v)]['msg'] = m

    def recvfrom(self, u, nodes):
        m = []
        for v in nodes:
            m.append(self.edges[(u, v)]['msg'])

        f_update = self.node[u]['u_func']
        x = self.get_repr(u)
        x_new = f_update(x, m)

        self.set_repr(u, x_new)

    def update_by_edge(self, e):
        u, v = e
        f_msg = self.edges[(u, v)]['m_func']
        m = f_msg(self.get_repr(u))
        f_update = self.node[u]['u_func']
        x = self.get_repr(u)
        x_new = f_update(x, m)
        x_new = self.node[u]['u_func']

        self.set_repr(u, x_new)

    def update_to(self, u):
        """Pull messages from 1-step away neighbors of u"""
        assert u in self.nodes

        msgs = []
        for v in self.pred[u]:
            f_msg = self.edges[(u, v)]['m_func']
            msgs.append(f_msg(self.get_repr(v)))

        f_update = self.node[u]['u_func']
        x = self.get_repr(u)
        x_new = f_update(x, msgs)
        x_new = self.node[u]['u_func']

        self.set_repr(u, x_new)

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

    def set_msg_func(self, func=None, u=None):
        """Function that gathers messages from neighbors"""
        def _default_msg_func(u):
            assert u in self.nodes
            msg_gathered = []
            for v in self.pred[u]:
                x = self.get_repr(v)
                if x is not None:
                    msg_gathered.append(x)
            return self._reduction_func(msg_gathered)

        # TODO: per node message function
        # TODO: 'sum' should be a separate function
        if func == None:
            self._msg_func = _default_msg_func
        else:
            self._msg_func = func

    def set_update_func(self, func=None, u=None):
        """
        Update function upon receiving an aggregate
        message from a node's neighbor
        """
        def _default_update_func(x, m):
            return x + m

        # TODO: per node update function
        if func == None:
            self._update_func = _default_update_func
        else:
            self._update_func = func

    def set_readout_func(self, func=None):
        """Readout function of the whole graph"""
        def _default_readout_func():
            valid_hs = []
            for x in self.nodes:
                h = self.get_repr(x)
                if h is not None:
                    valid_hs.append(h)
            return self._reduction_func(valid_hs)
#
        if func == None:
            self.readout_func = _default_readout_func
        else:
            self.readout_func = func

    def print_all(self):
        for n in self.nodes:
            print(n, self.nodes[n])
        print()

if __name__ == '__main__':
    import torch as th
    import torch.nn.functional as F
    import torch.nn as nn
    from torch.autograd import Variable as Var

    th.random.manual_seed(0)

    ''': this makes a digraph with double edges
    tg = nx.path_graph(10)
    g = mx_Graph(tg)
    g.print_all()

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

    '''
    g.set_update_func(nn.GRUCell(4, 4))
    for n in g:
        g.set_repr(n, Var(th.rand(2, 4)))

    print("\t**before:"); g.print_all()
    g.update_from(0)
    g.update_from(1)
    print("\t**after:"); g.print_all()

    print("\ntesting fwd update")
    g.clear()
    g.add_path([0, 1, 2])
    g.init_reprs()

    fwd_net = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    g.set_update_func(fwd_net)

    g.set_repr(0, Var(th.rand(2, 4)))
    print("\t**before:"); g.print_all()
    g.update_from(0)
    g.update_from(1)
    print("\t**after:"); g.print_all()
    '''
