#!/usr/bin/env python

"""
    children.py
"""

from __future__ import division

import sys
import torch
from tqdm import tqdm
from collections import Counter

from workers.helpers import InvalidGraphException

from basenet.helpers import to_numpy

class LoopyDataloader(object):
    """ Wrapper so we can use torchvision loaders in an infinite loop """
    def __init__(self, gen):
        self.batches_per_epoch = len(gen)
        self.epoch_batches     = 0
        self.epochs            = 0
        self.progress          = 0
        
        self._loop = self.__make_loop(gen)
    
    def __make_loop(self, gen):
        while True:
            self.epoch_batches = 0
            for data,target in gen:
                yield data, target
                
                self.progress = self.epochs + (self.epoch_batches / self.batches_per_epoch)
                self.epoch_batches += 1
            
            self.epochs += 1
    
    def __next__(self):
        return next(self._loop)


class Child(object):
    """ Wraps BaseNet model to expose a nice API for ripenet """
    def __init__(self, worker, dataloaders, verbose=True):
        self.worker       = worker
        self.dataloaders  = dict([(k, LoopyDataloader(v)) for k,v in dataloaders.items()])
        self.records_seen = Counter()
        self.verbose      = verbose
        
    def train_paths(self, paths, n=1, mode='train'):
        self.worker.reset_pipes()
        
        loader = self.dataloaders[mode]
        gen = paths
        if self.verbose:
            gen = tqdm(gen, desc='Child.train_paths (%s)' % mode)
        
        correct, total = 0, 0
        for path in gen:
            self.worker.set_path(path)
            if self.worker.is_valid:
                for _ in range(n):
                    data, target = next(loader)
                    self.worker.set_progress(loader.progress)
                    output, loss = self.worker.train_batch(data, target)
                    
                    if self.verbose:
                        correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
                        total += data.shape[0]
                        gen.set_postfix({'acc' : correct / total, "loss" : loss})
                    
                    self.records_seen[mode] += data.shape[0]
    
    def eval_paths(self, paths, n=1, mode='val'):
        self.worker.reset_pipes()
        
        rewards = []
        
        loader = self.dataloaders[mode]
        gen = paths
        if self.verbose:
            gen = tqdm(gen, desc='Child.eval_paths (%s)' % mode)
        
        correct, total = 0, 0
        for path in gen:
            self.worker.set_path(path)
            if self.worker.is_valid:
                acc = 0
                for _ in range(n):
                    data, target = next(loader)
                    output, _ = self.worker.eval_batch(data, target)
                    
                    if self.verbose:
                        correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
                        total += data.shape[0]
                        gen.set_postfix({"acc" : correct / total})
                    
                    acc += (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
                    
                    self.records_seen[mode] += data.shape[0]
                    
            else:
                acc = -0.1
            
            rewards.append(acc / n)
        
        return torch.FloatTensor(rewards).view(-1, 1)

class LazyChild(Child):
    def train_paths(self, paths):
        pass


