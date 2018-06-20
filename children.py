#!/usr/bin/env python

"""
    children.py
"""

from __future__ import print_function, division

import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

from basenet import Metrics
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
    
    def _run_paths(self, paths, n, mode, batch_fn):
        self.worker.reset_pipes()
        
        rewards = []
        loader = self.dataloaders[mode]
        gen = paths
        if self.verbose:
            gen = tqdm(gen, desc='Child (%s) (%s)' % ('train' if self.worker.training else 'eval', mode))
        
        for path in gen:
            self.worker.set_path(path)
            assert self.worker.is_valid, 'not self.worker.is_valid'
            correct, total = 0, 0
            for _ in range(n):
                data, target = next(loader)
                self.worker.set_progress(loader.progress)
                
                loss, metrics = batch_fn(data, target, metric_fns=[Metrics.n_correct])
                
                correct += metrics[0]
                total   += target.shape[0]
                
                self.records_seen[mode] += data.shape[0]
                
                if self.verbose and rewards:
                    gen.set_postfix(**{
                        "mean_acc"  : float(np.mean(rewards)),
                        "max_acc"   : float(np.max(rewards)),
                    })
            
            rewards.append(correct / total)
        
        return torch.Tensor(rewards).view(-1, 1)
    
    def train_paths(self, paths, n=1, mode='train'):
        _ = self.worker.train()
        return self._run_paths(
            paths=paths,
            n=n,
            mode=mode,
            batch_fn=self.worker.train_batch,
        )
    
    def eval_paths(self, paths, n=1, mode='val'):
        _ = self.worker.eval()
        return self._run_paths(
            paths=paths,
            n=n,
            mode=mode,
            batch_fn=self.worker.eval_batch,
        )


class LazyChild(Child):
    def train_paths(self, paths):
        pass
