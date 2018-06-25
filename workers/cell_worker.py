#!/usr/bin/env python

"""
    pipenet.py
"""

from __future__ import print_function, division

import sys
import numpy as np
from dask import get
from functools import partial
from dask.optimize import cull
from collections import OrderedDict, defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet
from basenet.helpers import to_numpy

from .helpers import InvalidGraphException

# --
# Helper layers

class PipeModule(nn.Module):
    def __init__(self, needs_path=False):
        super().__init__()
        self.needs_path = needs_path
    
    def set_pipes(self, pipes):
        raise NotImplemented


class Accumulator(PipeModule):
    def __init__(self, agg_fn=torch.sum, name='noname'):
        super().__init__()
        
        self.agg_fn = agg_fn
        self.name = name
    
    def forward(self, parts):
        parts = [part for part in parts if part is not None]
        if len(parts) == 0:
            return None
        else:
            return self.agg_fn(torch.stack(parts), dim=0)
    
    def __repr__(self):
        return 'Accumulator(%s)' % self.name


class Flatten(PipeModule):
    def forward(self, x):
        return x.view(x.shape[0], -1)
        
    def __repr__(self):
        return "Flatten()"

# SHARED_BN = True
# print('******** SHARED_BN=%d ********' % SHARED_BN, file=sys.stderr)
# if not SHARED_BN:
#     class PipeBatchNorm2d(PipeModule):
#         def __init__(self, *args, **kwargs):
#             super().__init__(needs_path=True)
            
#             self.make_new_layer = lambda: nn.BatchNorm2d(*args, **kwargs).cuda()
            
#             self.layers = {}
#             self.active_path = None
        
#         def set_path(self, path, callback):
#             path_desc = str(tuple(path))
#             if path_desc not in self.layers:
#                 new_layer = self.make_new_layer()        # Create new layer
#                 self.layers[path_desc] = new_layer       # Add to dict
#                 self.add_module(path_desc, new_layer)    # Register
#                 callback(mode='bn', new_layer=new_layer)
            
#             self.active_path = path_desc
        
#         def forward(self, x):
#             raise Exception
#             # assert self.active_path is not None, "!! PipeBatchNorm2d: active_path is None"
#             # return self.layers[self.active_path](x)
# else:
#     class PipeBatchNorm2d(PipeModule):
#         def __init__(self, *args, **kwargs):
#             super().__init__(needs_path=True)
            
#             kwargs.update({
#                 "track_running_stats" : False,
#             })
#             self.layer = nn.BatchNorm2d(*args, **kwargs).cuda()
            
#         def set_path(self, path, callback):
#             pass
        
#         def forward(self, x):
#             raise Exception
#             # return self.layer(x)


class IdentityLayer(PipeModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(needs_path=(in_channels != out_channels))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if (in_channels != out_channels) or (stride != 1):
            # self.bn = PipeBatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=stride)
        else:
            self.conv = None
            self.bn = None
    
    def forward(self, x):
        if self.conv is not None:
            # return self.conv(self.bn(x))
            return self.conv(x)
        else:
            return x
    
    # def set_path(self, path, callback):
    #     if self.bn is not None:
    #         _ = self.bn.set_path(path, callback)
    
    def __repr__(self):
        return "IdentityLayer(%d -> %d)" % (self.in_channels, self.out_channels)


class NoopLayer(PipeModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
    
    def forward(self, x):
        out = Variable(torch.zeros(x.shape[0], self.out_channels, x.shape[2] / self.stride, x.shape[3] / self.stride))
        if x.is_cuda:
            out = out.cuda()
        
        return out
    
    def __repr__(self):
        return "NoopLayer(%d -> %d | stride=%d)" % (self.in_channels, self.out_channels, self.stride)


class BNConv2d(PipeModule):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(needs_path=True)
        
        # self.add_module('bn', PipeBatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, **kwargs))
    
    def forward(self, x):
        # return self.conv(self.relu(self.bn(x)))
        return self.conv(self.relu(x))
    
    # def set_path(self, path, callback):
    #     _ = self.bn.set_path(path, callback)
    
    def __repr__(self):
        return 'BN' + self.conv.__repr__()


class ReshapePool2d(PipeModule):
    def __init__(self, in_channels, out_channels, mode='avg', **kwargs):
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv = None
        
        self.mode = mode
        if mode == 'avg':
            self.pool = nn.AvgPool2d(**kwargs)
        else:
            self.pool = nn.MaxPool2d(**kwargs)
    
    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        
        return self.pool(x)
    
    def __repr__(self):
        return 'ReshapePool2d(mode=%s, in_channels=%d, out_channels=%d)' % (self.mode, self.in_channels, self.out_channels)


class BNSepConv2d(BNConv2d):
    def __init__(self, **kwargs):
        assert 'groups' not in kwargs, "BNSepConv2d: cannot specify groups"
        kwargs['groups'] = min(kwargs['in_channels'], kwargs['out_channels'])
        super().__init__(**kwargs)
    
    def __repr__(self):
        return 'BNSep' + self.conv.__repr__()


# --
# Blocks

class CellBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_nodes=2, num_branches=2):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        self.op_fns = OrderedDict([
            ("noop____", NoopLayer),
            ("identity", IdentityLayer),
            ("conv3___", partial(BNConv2d, kernel_size=3, padding=1)),
            ("conv5___", partial(BNConv2d, stride=stride, kernel_size=5, padding=2)),
            ("sepconv3", partial(BNSepConv2d, stride=stride, kernel_size=3, padding=1)),
            ("sepconv5", partial(BNSepConv2d, stride=stride, kernel_size=5, padding=2)),
            ("avgpool_", partial(ReshapePool2d, mode='avg', stride=stride, kernel_size=3, padding=1)),
            ("maxpool_", partial(ReshapePool2d, mode='max', stride=stride, kernel_size=3, padding=1)),
        ])
        self.op_lookup = dict(zip(range(len(self.op_fns)), self.op_fns.keys()))
        
        # --
        # Create nodes (just sum accumulators)
        
        self.nodes = OrderedDict([])
        for node_id in range(num_nodes):
            self.nodes['node_%d' % node_id] = Accumulator(name="node_%d" % node_id)
        
        for k, v in self.nodes.items():
            self.add_module(str(k), v)
        
        # --
        # Create pipes
        
        self.pipes = OrderedDict([])
        
        # Add pipes from input data to all nodes
        for trg_id in range(num_nodes):
            for src_id in ['data_0']:
                for branch in range(num_branches):
                    for op_key, op_fn in self.op_fns.items():
                        self.pipes[(src_id, 'node_%d' % trg_id, op_key, branch)] = op_fn(in_channels=in_channels, out_channels=out_channels, stride=stride)
        
        # Add pipes between all nodes
        for trg_id in range(num_nodes):
            for src_id in range(trg_id):
                for branch in range(num_branches):
                    for op_key, op_fn in self.op_fns.items():
                        self.pipes[('node_%d' % src_id, 'node_%d' % trg_id, op_key, branch)] = op_fn(in_channels=out_channels, out_channels=out_channels, stride=1)
        
        for k, v in self.pipes.items():
            self.add_module(str(k), v)
        
        # --
        # Set default architecture
        
        # 0002|0112
        self._default_pipes = [
            ('data_0', 'node_0', 'noop____', 0),
            ('data_0', 'node_0', 'conv3___', 1),
            ('data_0', 'node_1', 'identity', 0),
            ('node_0', 'node_1', 'conv3___', 1),
        ]
        self.reset_pipes()
    
    def reset_pipes(self):
        self.set_pipes(self._default_pipes)
    
    def get_pipes(self):
        return [pipe_name for pipe_name, pipe in self.pipes.items() if pipe.active]
    
    def get_pipes_mask(self):
        return [pipe.active for pipe in self.pipes.values()]
    
    def set_pipes(self, pipes):
        self.active_pipes = [tuple(pipe) for pipe in pipes]
        
        for pipe_name, pipe in self.pipes.items():
            pipe.active = pipe_name in self.active_pipes
        
        # --
        # Add cells to graph
        
        self.graph = OrderedDict([('data_0', None)])
        for node_name, node in self.nodes.items():
            node_inputs = [pipe_name for pipe_name, pipe in self.pipes.items() if (pipe_name[1] == node_name) and (pipe.active)]
            self.graph[node_name] = (node, node_inputs)
        
        # --
        # Add pipes to graph
        
        for pipe_name, pipe in self.pipes.items():
            if pipe.active:
                self.graph[pipe_name] = (pipe, pipe_name[0])
        
        # --
        # Gather loose ends for output
        
        nodes_w_output  = set([k[0] for k in self.graph.keys() if isinstance(k, tuple)])
        nodes_wo_output = [k for k in self.graph.keys() if ('node' in k) and (k not in nodes_w_output)]
        
        self.graph['_output'] = (Accumulator(name='_output', agg_fn=torch.mean), nodes_wo_output) # sum or avg or concat?  which is best?
    
    def set_path(self, path, callback):
        path = to_numpy(path)
        
        for child in self.children():
            if child.needs_path:
                child.set_path(path, callback)
        
        pipes = []
        for i, path_block in enumerate(path.reshape(-1, 4)):
            trg_id = 'node_%d' % i # !! Indexing here is a little confusing
            
            src_0 = 'node_%d' % (path_block[0] - 1) if path_block[0] != 0 else "data_0" # !! Indexing here is a little confusing
            src_1 = 'node_%d' % (path_block[1] - 1) if path_block[1] != 0 else "data_0" # !! Indexing here is a little confusing
            
            pipes += [
                (src_0, trg_id, self.op_lookup[path_block[2]], 0),
                (src_1, trg_id, self.op_lookup[path_block[3]], 1),
            ]
        
        for pipe in pipes:
            assert pipe in self.pipes.keys()
        
        self.set_pipes(pipes)
    
    def trim_pipes(self):
        for k,v in self.pipes.items():
            if k not in self.active_pipes:
                delattr(self, str(k))
    
    @property
    def is_valid(self, layer='_output'):
        return 'data_0' in cull(self.graph, layer)[0]
    
    def forward(self, x, layer='_output'):
        if not self.is_valid:
            raise InvalidGraphException
        
        self.graph['data_0'] = x
        out = get(self.graph, layer)
        self.graph['data_0'] = None
        
        if out is None:
            raise InvalidGraphException
        
        return out

# --
# Models

class _CellWorker(BaseNet):
    def reset_pipes(self):
        for cell_block in self.cell_blocks:
            _ = cell_block.reset_pipes()
    
    def get_pipes(self):
        tmp = []
        for cell_block in self.cell_blocks:
            tmp.append(cell_block.get_pipes())
        
        return tmp
    
    def get_pipes_mask(self):
        tmp = []
        for cell_block in self.cell_blocks:
            tmp.append(cell_block.get_pipes_mask())
        
        return tmp
    
    def set_pipes(self, pipes):
        for cell_block in self.cell_blocks:
            _ = cell_block.set_pipes(pipes)
    
    def set_path_callback(self, mode, **kwargs):
        if mode == 'bn':
            self.opt.add_param_group({
                "params" : kwargs['new_layer'].parameters(),
            })
        else:
            raise Exception("_CellWorker.set_path_callback: unknown mode %s" % mode, file=sys.stderr)
    
    def set_path(self, path):
        for cell_block in self.cell_blocks:
            cell_block.set_path(path, self.set_path_callback)
    
    def trim_pipes(self):
        for cell_block in self.cell_blocks:
            _ = cell_block.trim_pipes()
    
    @property
    def is_valid(self, layer='_output'):
        return np.all([cell_block.is_valid for cell_block in self.cell_blocks])


class CellWorker(_CellWorker):
    pass
#     def __init__(self, num_classes=10, input_channels=3, num_blocks=[2, 2, 2, 2], num_channels=[32, 64, 128, 256, 512], num_nodes=2):
#         super().__init__()
        
#         self.num_nodes = num_nodes
        
#         self.prep = nn.Conv2d(in_channels=input_channels, out_channels=num_channels[0], kernel_size=3, padding=1)
        
#         self.cell_blocks = []
        
#         all_layers = []
#         for i, (block, in_channels, out_channels) in enumerate(zip(num_blocks, num_channels[:-1], num_channels[1:])):
#             layers = []
            
#             # Add cell at beginning that changes num channels
#             cell_block = CellBlock(in_channels=in_channels, out_channels=out_channels, num_nodes=num_nodes, stride=2 if i > 0 else 1)
#             layers.append(cell_block)
#             self.cell_blocks.append(cell_block)
            
#             # Add cells that preserve channels
#             for _ in range(block - 1):
#                 cell_block = CellBlock(in_channels=out_channels, out_channels=out_channels, num_nodes=num_nodes , stride=1)
#                 layers.append(cell_block)
#                 self.cell_blocks.append(cell_block)
            
#             all_layers.append(nn.Sequential(*layers))
        
#         self.layers = nn.Sequential(*all_layers)
        
#         self.classifier = nn.Linear(num_channels[-1], num_classes)
    
#     def forward(self, x):
#         x = self.prep(x)
#         x = self.layers(x)
        
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
        
#         return x

from .layers import AdaptiveMultiPool2d, Flatten
class BoltWorker(_CellWorker):
    def __init__(self, num_features=512, num_classes=200, num_nodes=2, num_branches=2):
        super().__init__()
        
        self.cell_blocks = [
            CellBlock(
                in_channels=num_features,
                out_channels=num_features,
                num_nodes=num_nodes,
                num_branches=num_branches
            ),
        ]
        
        self.layers = nn.Sequential(*self.cell_blocks)
        
        self.classifier = nn.Sequential(*[
            AdaptiveMultiPool2d(output_size=(1, 1)),
            Flatten(),
            nn.BatchNorm1d(2 * num_features),
            nn.Linear(in_features=2 * num_features, out_features=num_classes),
        ])
    
    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x