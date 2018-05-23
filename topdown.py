import networkx as nx
from mx import mx_Graph
from glimpse import create_glimpse
import torch as T
import torch.nn as nn
import torch.functional as F
import torchvision.models as MODELS
import torch.nn.init as INIT

def dfs_walk(tree, curr, l):
    if len(tree.succ[curr]) == 0:
        return
    else:
        for n in tree.succ[curr]:
            l.append((curr, n))
            dfs_walk(tree, n, l)
            l.append((n, curr))

def build_cnn(**config):
    cnn_list = []
    filters = config['filters']
    kernel_size = config['kernel_size']
    in_channels = config.get('in_channels', 3)
    final_pool_size = config['final_pool_size']

    for i in range(len(filters)):
        module = nn.Conv2d(
            in_channels if i == 0 else filters[i-1],
            filters[i],
            kernel_size,
            padding=tuple((_ - 1) // 2 for _ in kernel_size),
            )
        INIT.xavier_uniform_(module.weight)
        INIT.constant_(module.bias, 0)
        cnn_list.append(module)
        if i < len(filters) - 1:
            cnn_list.append(nn.LeakyReLU())
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)

def build_resnet_cnn(**config):
    n_layers = config['n_layers']
    final_pool_size = config['final_pool_size']

    resnet = MODELS.resnet18(pretrained=True)
    cnn_list = list(resnet.children())[0:n_layers]
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)

class MessageModule(nn.Module):
    def forward(self, state):
        h, b, y = state
        return h

class UpdateModule(nn.Module):
    """
    UpdateModule:

    Returns:
        h: new state
        b: new bounding box
        a: attention (for readout)
        y: prediction
    """
    def __init__(self, **config):
                 #h_dims=128,
                 #n_classes=10,
                 #steps=5,
                 #filters=[16, 32, 64, 128, 256],
                 #kernel_size=(3, 3),
                 #final_pool_size=(2, 2),
                 #glimpse_type='gaussian',
                 #glimpse_size=(15, 15),
                 #cnn='resnet'
                 #):
        super(UpdateModule, self).__init__()
        glimpse_type = config['glimpse_type']
        glimpse_size = config['glimpse_size']
        self.glimpse = create_glimpse(glimpse_type, glimpse_size)

        cnn = config['cnn']
        final_pool_size = config['final_pool_size']
        if cnn == 'resnet':
            n_layers = config['n_layers']
            self.cnn_resnet = build_resnet_cnn(
                    n_layers=n_layers,
                    final_pool_size=final_pool_size,
                    )
        else:
            filters = config['filters']
            kernel_size = config['kernel_size']
            self.cnn = build_cnn(
                    filters=filters,
                    kernel_size=kernel_size,
                    final_pool_size=final_pool_size,
                    )

        h_dims = config['h_dims']
        n_classes = config['n_classes']
        self.net_b = nn.Linear(h_dims, self.glimpse.att_params)
        self.net_y = nn.Linear(h_dims, n_classes)
        self.net_a = nn.Linear(h_dims, 1)

    def set_image(self, x):
        self.x = x

    def forward(self, node_state, message):
        h, b, a, y = node_state
        b_new = b + self.net_b(h)
        y_new = y + self.net_y(h)
        a_new = self.net_a(h)

        g = self.glimpse(self.x, b_new)
        h_new = h + self.cnn(g) + message

        return h_new, b_new, a_new, y_new

def update_local():
    pass

class ReadoutModule(nn.Module):
    def __init__(self, *args, **kwarg):
        super(ReadoutModule, self).__init__()
        self.y = nn.Linear(kwarg['h_dims'], kwarg['n_classes'])

    def forward(self, nodes_state):
        h, _, a, _ = nodes_state
        b_of_h = T.sum(a * h)
        y = nn.ReLU(self.y(b_of_h))
        return y

class DFSGlimpseClassifier(nn.Module):
    def __init__(self,
                 h_dims=128,
                 n_classes=10,
                 steps=5,
                 filters=[16, 32, 64, 128, 256],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 glimpse_type='gaussian',
                 glimpse_size=(15, 15),
                 cnn='resnet'
                 ):
        nn.Module.__init__(self)

        #self.T_MAX_RECUR = kwarg['steps']
        self.T_MAX_RECUR = 1

        t = nx.balanced_tree(2, 2)
        t_uni = nx.bfs_tree(t, 0)
        self.G = mx_Graph(t)
        self.root = 0

        self.message_module = MessageModule()
        self.G.register_message_func(self.message_module) # default: just copy

        #self.update_module = UpdateModule(h_dims, n_classes, glimpse_size)
        self.update_module = UpdateModule(
            glimpse_type=glimpse_type,
            glimpse_size=glimpse_size,
            n_layers=6,
            h_dims=h_dims,
            n_classes=n_classes,
            final_pool_size=final_pool_size,
            cnn=cnn
        )
        self.G.register_update_func(self.update_module)

        self.readout_module = ReadoutModule(h_dims=h_dims, n_classes=n_classes)
        self.G.register_readout_func(self.readout_module)

        self.walk_list = []
        dfs_walk(t_uni, self.root, self.walk_list)

    def forward(self, x):
        self.update_module.set_image(x)

        pred = []
        #TODO: the following two lines is needed for single object
        #TODO: but not useful or wrong for multi-obj
        self.G.recvfrom(self.root, [])
        pred.append(self.G.readout(self.root))

        for u, v in self.walk_list:
            self.G.update_by_edge((v, u))
            for i in self.T_MAX_RECUR:
                self.G.update_local(u)
            y = self.G.readout()
            pred.append(y)
        return pred

if __name__ == "__main__":

    model = DFSGlimpseClassifier()
