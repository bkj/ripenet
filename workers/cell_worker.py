#!/usr/bin/env python

"""
    pipenet.py
"""

from __future__ import print_function, division

import sys
import itertools
import numpy as np
from dask import get
from dask.optimization import cull
from pprint import pprint
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import basenet
from basenet.helpers import to_numpy

from .helpers import InvalidGraphException

# --
# Helper layers

class Accumulator(nn.Module):
    def __init__(self, agg_fn=torch.sum, name='noname'):
        super(Accumulator, self).__init__()
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


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x
    
    def __repr__(self):
        return "IdentityLayer()"


class ZeroLayer(nn.Module):
    def forward(self, x):
        return None
    
    def __repr__(self):
        return "ZeroLayer()"


class BNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BNConv2d, self).__init__()
        
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, **kwargs))
    
    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))
    
    def __repr__(self):
        return 'BN' + self.conv.__repr__()

class BNSepConv2d(nn.Module):
    def __init__(self, **kwargs):
        assert 'groups' not in kwargs, "BNSepConv2d: cannot specify groups"
        super(BNSepConv2d, self).__init__(**kwargs)
    
    def __repr__(self):
        return 'BNSep' + self.conv.__repr__()


# --
# Blocks

class CellBlock(nn.Module):
    def __init__(self, channels, stride=1, num_nodes=2):
        super(CellBlock, self).__init__()
        
        self.op_fns = OrderedDict([
            ("identity", IdentityLayer),
            ("zero____", ZeroLayer),
            ("conv3___", partial(BNConv2d, in_channels=channels, out_channels=channels, stride=stride, kernel_size=3, padding=1)),
            ("conv5___", partial(BNConv2d, in_channels=channels, out_channels=channels, stride=stride, kernel_size=5, padding=2)),
            ("sepconv3", partial(BNSepConv2d, in_channels=channels, out_channels=channels, stride=stride, kernel_size=3, padding=1)), # depthwise separable
            ("sepconv5", partial(BNSepConv2d, in_channels=channels, out_channels=channels, stride=stride, kernel_size=5, padding=2)), # depthwise separable
            ("maxpool_", partial(nn.MaxPool2d, stride=stride, kernel_size=3, padding=1)),
            ("avgpool_", partial(nn.AvgPool2d, stride=stride, kernel_size=3, padding=1)),
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
        num_branches = 2
        
        # !! Should do [data_0, node_1, ..., node_(b+1)] instead
        
        # Add pipes from input data to all nodes
        for trg_id in range(num_nodes):
            for src_id in ['data_0']:
                for branch in range(num_branches):
                    for op_key, op_fn in self.op_fns.items():
                        self.pipes[(src_id, 'node_%d' % trg_id, op_key, branch)] = op_fn()
        
        # Add pipes between all nodes
        for trg_id in range(num_nodes):
            for src_id in range(trg_id):
                for branch in range(num_branches):
                    for op_key, op_fn in self.op_fns.items():
                        self.pipes[('node_%d' % src_id, 'node_%d' % trg_id, op_key, branch)] = op_fn()
        
        
        for k, v in self.pipes.items():
            self.add_module(str(k), v)
        
        # --
        # Set default architecture
        
        self._default_pipes = [
            ('data_0', 'node_0', 'conv3___', 0),
            ('node_0', 'node_1', 'conv3___', 0),
            ('data_0', 'node_1', 'identity', 1),
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
        
        self.graph = OrderedDict([('data_0',   None)])
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
        
        self.graph['_output'] = (Accumulator(name='_output'), nodes_wo_output) # May want to sum/avg/concat
    
    def set_path(self, path):
        path = to_numpy(path).reshape(-1, 4)
        pipes = []
        for i, path_block in enumerate(path):
            trg_id = 'node_%d' % i # !! Indexing here is a little bad
            
            src_0 = 'node_%d' % (path_block[0] - 1) if path_block[0] != 0 else "data_0" # !! Indexing here is a little bad
            src_1 = 'node_%d' % (path_block[1] - 1) if path_block[1] != 0 else "data_0" # !! Indexing here is a little bad
            
            pipes += [
                (src_0, trg_id, self.op_lookup[path_block[2]], 0),
                (src_1, trg_id, self.op_lookup[path_block[3]], 1),
            ]
        
        for pipe in pipes:
            assert pipe in self.pipes.keys()
        
        self.set_pipes(pipes)
    
    @property
    def is_valid(self, layer='_output'):
        # !! I think everything will be valid, because we average loose ends
        return 'data_0' in cull(self.graph, layer)[0]
    
    def forward(self, x, layer='_output'):
        if not self.is_valid:
            raise InvalidGraphException
        
        self.graph['data_0'] = x
        out = get(self.graph, layer)
        self.graph['data_0'] = None
        return out

# --
# Models

class CellWorker(basenet.BaseNet):
    
    def __init__(self, num_classes=10, num_blocks=[2, 2, 2, 2], num_channels=[64, 128, 256, 512]):
        super(CellWorker, self).__init__()
        
        self.prep = BNConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        
        self.cell_blocks = []
        
        all_layers = []
        for i, (block, channels) in enumerate(zip(num_blocks, num_channels)):
            layers = []
            for _ in range(block):
                cell_block = CellBlock(channels=channels)
                layers.append(cell_block)
                self.cell_blocks.append(cell_block)
            
            all_layers.append(nn.Sequential(*layers))
            
            if (i + 1) < len(num_blocks):
                all_layers.append(BNConv2d(in_channels=channels, out_channels=num_channels[i + 1], kernel_size=3, padding=1, stride=2))
        
        self.layers = nn.Sequential(*all_layers)
        self.classifier = nn.Sequential(
            BNConv2d(in_channels=num_channels[-1], out_channels=num_classes, kernel_size=1, padding=0, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    
    def forward(self, x):
        x = self.prep(x)
        x = self.layers(x)
        x = self.classifier(x)
        x = x.view((x.shape[0], x.shape[1]))
        return x
    
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
    
    def set_path(self, path):
        for cell_block in self.cell_blocks:
            _ = cell_block.set_path(path)
    
    @property
    def is_valid(self, layer='_output'):
        return np.all([cell_block.is_valid for cell_block in self.cell_blocks])
