import networkx as nx
from mx import mx_Graph
from glimpse import create_glimpse
import torch as T
import torch.nn as nn
import torch.functional as F
import torchvision.models as MODELS
import torch.nn.init as INIT
from util import USE_CUDA
import numpy as np
import skorch

batch_size = 32

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
        h, b, a, y = state
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

        h_dims = config['h_dims']
        n_classes = config['n_classes']
        self.net_i = nn.Sequential(
                nn.Linear(h_dims, h_dims),
                nn.ReLU(),
                nn.Linear(h_dims, h_dims),
                nn.ReLU(),
                )
        self.net_b = nn.Linear(h_dims, self.glimpse.att_params)
        self.net_y = nn.Linear(h_dims, n_classes)
        self.net_a = nn.Linear(h_dims, 1)

        self.h_to_h = nn.GRUCell(h_dims * 2, h_dims)

        cnn = config['cnn']
        final_pool_size = config['final_pool_size']
        if cnn == 'resnet':
            n_layers = config['n_layers']
            self.cnn = build_resnet_cnn(
                    n_layers=n_layers,
                    final_pool_size=final_pool_size,
                    )
            self.net_h = nn.Linear(128 * np.prod(final_pool_size), h_dims)
        else:
            filters = config['filters']
            kernel_size = config['kernel_size']
            self.cnn = build_cnn(
                    filters=filters,
                    kernel_size=kernel_size,
                    final_pool_size=final_pool_size,
                    )
            self.net_h = nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims)

        self.max_recur = config.get('max_recur', 1)

    def set_image(self, x):
        self.x = x

    def forward(self, node_state, message):
        h, b, a, y = node_state
        batch_size = h.shape[0]
        message_avg = 0 if len(message) == 0 else T.stack(message).mean(0)
        h_new = h
        b_new = b

        for i in range(self.max_recur):
            g = self.glimpse(self.x, b_new[:, None])[:, 0]
            h_in = T.cat([self.net_h(self.cnn(g).view(batch_size, -1)), message_avg], -1)
            h_new = self.h_to_h(h_in, h_new)

            i_new = self.net_i(h_new)
            b_new = b + self.net_b(i_new)
            y_new = y + self.net_y(i_new)
            a_new = self.net_a(i_new)

        return h_new, b_new, a_new, y_new

def update_local():
    pass

class ReadoutModule(nn.Module):
    '''
    Returns the logits of classes
    '''
    def __init__(self, *args, **kwarg):
        super(ReadoutModule, self).__init__()
        self.y = nn.Linear(kwarg['h_dims'], kwarg['n_classes'])

    def forward(self, nodes_state):
        h = T.stack([s[0] for s in nodes_state], 1)
        a = T.stack([s[2] for s in nodes_state], 1)
        b_of_h = T.sum(a * h, 1)
        y = self.y(b_of_h)
        return y

class DFSGlimpseSingleObjectClassifier(nn.Module):
    def __init__(self,
                 h_dims=128,
                 n_classes=10,
                 filters=[16, 32, 64, 128, 256],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 glimpse_type='gaussian',
                 glimpse_size=(15, 15),
                 cnn='resnet'
                 ):
        nn.Module.__init__(self)

        #self.T_MAX_RECUR = kwarg['steps']

        t = nx.balanced_tree(2, 2)
        t_uni = nx.bfs_tree(t, 0)
        self.G = mx_Graph(t)
        self.root = 0
        self.h_dims = h_dims
        self.n_classes = n_classes

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
            cnn=cnn,
            max_recur=1,    # T_MAX_RECUR
        )
        self.G.register_update_func(self.update_module)

        self.readout_module = ReadoutModule(h_dims=h_dims, n_classes=n_classes)
        self.G.register_readout_func(self.readout_module)

        self.walk_list = []
        dfs_walk(t_uni, self.root, self.walk_list)

    def forward(self, x):
        batch_size = x.shape[0]

        self.update_module.set_image(x)
        self.G.init_reprs((
            x.new(batch_size, self.h_dims).zero_(),
            x.new(batch_size, self.update_module.glimpse.att_params).zero_(),
            x.new(batch_size, 1).zero_(),
            x.new(batch_size, self.n_classes).zero_(),
            ))

        #TODO: the following two lines is needed for single object
        #TODO: but not useful or wrong for multi-obj
        self.G.recvfrom(self.root, [])

        for u, v in self.walk_list:
            self.G.update_by_edge((v, u))
            # update local should be inside the update module
            #for i in self.T_MAX_RECUR:
            #    self.G.update_local(u)
        return self.G.readout()


class Net(skorch.NeuralNet):
    def initialize_criterion(self):
        # Overriding this method to skip initializing criterion as we don't use it.
        pass

    def initialize_optimizer(self):
        self.opt = self.opt_class(self.module_.parameters(), self.lr)

    def get_split_datasets(self, X, y=None, **fit_params):
        # Overriding this method to use our own dataloader to change the X
        # in signature to (train_dataset, valid_dataset)
        X_train, X_valid = X
        train = self.get_dataset(X_train, None)
        valid = self.get_dataset(X_valid, None)
        return train, valid

    def train_step(self, Xi, yi, **fit_params):
        step = skorch.NeuralNet.train_step(self, Xi, yi, **fit_params)
        loss = step['loss']
        y_pred = step['y_pred']
        valid_loss = self.get_loss(y_pred, y_true, training=False)
        return {
                'loss': loss,
                'y_pred': y_pred,
                'valid_loss': valid_loss,
                }

    def get_loss(self, y_pred, y_true, X=None, training=False):
        if training:
            return F.cross_entropy(y_pred, y_true)
        else:
            return (y_pred.max(1)[1] == y_true).sum()


class Dump(skorch.callbacks.Callback):
    def initialize(self):
        self.epoch = 0
        self.batch = 0
        self.correct = 0
        self.total = 0
        return self

    def on_epoch_begin(self, net, **kwargs):
        self.epoch += 1
        self.batch = 0
        self.correct = 0
        self.total = 0

    def on_batch_end(self, net, **kwargs):
        self.batch += 1
        if kwargs['training']:
            print('#', self.epoch, self.batch, kwargs['loss'], kwargs['valid_loss'])
        else:
            self.correct += kwargs['valid_loss']
            self.total += kwargs['X'].shape[0]

    def on_epoch_end(self, net, **kwargs):
        print('@', self.epoch, self.correct, '/', self.total)


def data_generator(dataloader, batch_size):
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        yield x, y


if __name__ == "__main__":
    from datasets import MNISTMulti
    from torch.utils.data import DataLoader

    mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=50, image_cols=50, download=True)
    mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=50, image_cols=50, download=False, mode='valid')

    mnist_train_dl = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    mnist_valid_dl = DataLoader(mnist_valid, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    net = Net(
            module=model,
            max_epochs=50,
            optimizer=T.optim.RMSprop,
            lr=1e-4,
            batch_size=batch_size,
            device='cuda' if USE_CUDA else 'cpu',
            callbacks=[
                skorch.callbacks.Checkpoint(),
                skorch.callbacks.GradientNormClipping(1),
                Dump(),
                ],
            iterator_train=data_generator,
            iterator_train__dataloader=mnist_train_dl,
            iterator_valid=data_generator,
            iterator_valid__dataloader=mnist_valid_dl,
            )

    net.fit()
