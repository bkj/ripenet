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

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet
from basenet.helpers import to_numpy

from .helpers import InvalidGraphException

# --
# Blocks

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


class PBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, active=True, verbose=False):
        super(PBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))
        
        self.active    = active
        self.verbose   = verbose
        self.in_planes = in_planes
        self.planes    = planes
        self.stride    = stride
    
    def forward(self, x):
        if self.active and (x is not None):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            return out + shortcut
        else:
            return None
    
    def __repr__(self):
        return 'PBlock(%d -> %d | stride=%d | active=%d)' % (self.in_planes, self.planes, self.stride, self.active)


class CBlock(nn.Module):
    """ same as PBlock, but w/o batchnorm or activations """
    def __init__(self, in_planes, planes, stride=1, active=True, verbose=False):
        super(CBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.active    = active
        self.in_planes = in_planes
        self.planes    = planes
        self.stride    = stride
    
    def forward(self, x):
        if self.active and (x is not None):
            return self.conv(x)
        else:
            return None
    
    def __repr__(self):
        return 'CBlock(%d -> %d | stride=%d | active=%d)' % (self.in_planes, self.planes, self.stride, self.active)

# --
# Network

class MaskWorker(BaseNet):
    def __init__(self, num_blocks=[2, 2, 2, 2], lr_scheduler=None, num_classes=10, **kwargs):
        super(MaskWorker, self).__init__(**kwargs)
        
        # --
        # Preprocessing
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # --
        # Construct dense graph
        
        self._graph_entrypoint = 2 ** 6
        cell_sizes = [int(c) for c in 2 ** np.arange(6, 10)]
        
        self.cells = OrderedDict([])
        for cell_size in cell_sizes:
            self.cells[cell_size] = PBlock(cell_size, cell_size, stride=1)
        
        self.pipes = OrderedDict([])
        for cell_size_0, cell_size_1 in itertools.combinations(cell_sizes, 2):
            self.pipes[(cell_size_0, cell_size_1, 0)] = PBlock(cell_size_0, cell_size_1, stride=int(cell_size_1 / cell_size_0))
            self.pipes[(cell_size_0, cell_size_1, 1)] = CBlock(cell_size_0, cell_size_1, stride=int(cell_size_1 / cell_size_0))
        
        for k, v in self.cells.items():
            self.add_module(str(k), v)
        
        for k, v in self.pipes.items():
            self.add_module(str(k), v)
        
        self.pipe_names = list(self.pipes.keys())
        
        # --
        # Classifier
        
        self.linear = nn.Linear(512, num_classes)
        
        # --
        # Set default pipes
        
        self._default_pipes = list(zip(cell_sizes[:-1], cell_sizes[1:], [0] * (len(cell_sizes) - 1)))
        self.reset_pipes()
    
    def reset_pipes(self):
        self.set_pipes(self._default_pipes)
    
    def get_pipes(self):
        return [pipe_name for pipe_name, pipe in self.pipes.items() if pipe.active]
    
    def get_pipes_mask(self):
        return [pipe.active for pipe in self.pipes.values()]
    
    def set_path(self, mask):
        mask = to_numpy(mask) == 1
        self.set_pipes(np.array(self.pipe_names)[mask])
    
    def set_pipes(self, pipes):
        self.active_pipes = [tuple(pipe) for pipe in pipes]
        
        for pipe_name, pipe in self.pipes.items():
            pipe.active = pipe_name in self.active_pipes
        
        # Create cells + accumulators
        self.graph = {'_graph_input' :  None}
        for cell_name, cell in self.cells.items():
            if cell_name != self._graph_entrypoint:
                acc_name = '%d_acc' % cell_name
                
                # Determine active inputs
                in_pipes = filter(lambda p: p[1] == cell_name, self.pipe_names)
                in_pipes = filter(lambda p: self.pipes[p].active, in_pipes)
                in_pipes = list(in_pipes)
                
                self.graph[cell_name] = (cell, acc_name)
                self.graph[acc_name]  = (Accumulator(name=acc_name), in_pipes)
            else:
                self.graph[cell_name] = (cell, '_graph_input')
        
        # Create pipes
        for pipe_name, pipe in self.pipes.items():
            self.graph[pipe_name] = (pipe, pipe_name[0])
    
    @property
    def is_valid(self, layer=512):
        return '_graph_input' in cull(self.graph, layer)[0]
    
    def forward(self, x, layer=512):
        if not self.is_valid:
            raise InvalidGraphException
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        self.graph['_graph_input'] = out
        out = get(self.graph, layer)
        self.graph['_graph_input'] = None
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
