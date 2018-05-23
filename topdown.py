import networkx as nx
from mx import mx_Graph
from glimpse import create_glimpse
import torch as T
import torch.nn as nn
import torch.functional as F
import torchvision.models as MODELS
import torch.nn.init as INIT

t = nx.balanced_tree(2, 2)
t_uni = nx.bfs_tree(t, 0)

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
    num_layers = config['layers']
    final_pool_size = config['final_pool_size']

    resnet = MODELS.resnet18(pretrained=True)
    cnn_list = list(resnet.children())[0:num_layers]
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)

class ResNet(nn.Module):
    def __init__(self, x_shape=None):
        super(ResNet, self).__init__()
        resnet = MODELS.resnet18(pretrained=True)
        modules = list(resnet.children())[0:6]
        self.resnet = nn.Sequential(*modules)
        if x_shape is not None:
            x = T.rand(x_shape)
            x = self.resnet(x)
            shape = T.tensor(x.shape)
            fc_in = T.prod(shape[1:])
            self.fc = nn.Linear(fc_in, 10)

    def forward(self, x):
        x = self.resnet(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class MessageModule(nn.Module):
    def forward(self, state):
        h, b, y = state
        return h

class UpdateModule(nn.Module):
    '''UpdateModule:
    '''
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
        #h_dims=kwarg['h_dims'],
        #n_classes=kwarg['n_classes'],
        #glimpse_size=kwarg['glimpse_size'],

        nn.Module.__init__(self)
        t = nx.balanced_tree(2, 2)
        t_uni = nx.bfs_tree(t, 0)

        self.message_module = MessageModule()
        #self.update_module = UpdateModule(h_dims, n_classes, glimpse_size)
        self.update_module = UpdateModule()

        self.G = mx_Graph.DiGraph(t)
        self.G.register_update_module(self.update_module)
        #self.G.register_message_module(self.message_module) # default: just copy
        #self.G.register_readout_module(nn.Linear(h_dims, n_classes))

        root = 0
        self.e_list = []
        dfs_walk(t_uni, root, self.e_list)

    def forward(self, x):
        self.update_module.set_image(x)
        pred = []
        root = 0
        T_MAX_RECUR = 1
        y = self.classifier(self.G.get_repr(root))
        pred.append(y)
        for u, v in self.e_list:
            self.G.update_by_edge((v, u))
            for i in T_MAX_RECUR:
                self.G.update_local(u)
            y = self.classifier(self.G.get_repr(u))
            pred.append(y)
        return pred

#model = mx_Graph(t)
#model.register_update_func()

model = DFSGlimpseClassifier()

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

