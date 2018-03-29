#!/usr/bin/env python

"""
    eval.py
"""

import os
import sys
import json
import atexit
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from rsub import *
from matplotlib import pyplot as plt

from children import Child
from workers import CellWorker
from data import make_cifar_dataloaders

from basenet.helpers import to_numpy, set_seeds
from basenet.lr import LRSchedule

np.set_printoptions(linewidth=120)

# --
# CLI

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
                output, loss = self.worker.eval_batch(data, target)
                
                if self.verbose:
                    correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
                    total += data.shape[0]
                    gen.set_postfix({'acc' : correct / total, "loss" : loss})
                
                acc += (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
                
                self.records_seen[mode] += data.shape[0]
                
        else:
            acc = -0.1
            raise Exception('not self.worker.is_valid')
        
        rewards.append(acc / n)
    
    return torch.FloatTensor(rewards).view(-1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num-ops', type=int, default=6)   # Number of ops to sample
    parser.add_argument('--num-nodes', type=int, default=3) # Number of cells to sample
        
    parser.add_argument('--child-lr-init', type=float, default=0.1)
    parser.add_argument('--child-lr-schedule', type=str, default='constant')
    parser.add_argument('--child-lr-epochs', type=int, default=1000) # For LR schedule
    parser.add_argument('--child-sgdr-period-length', type=float, default=10)
    parser.add_argument('--child-sgdr-t-mult',  type=float, default=2)
    
    parser.add_argument('--train-size', type=float, default=0.9)     # Proportion of training data to use for training (vs validation) 
    parser.add_argument('--pretrained-path', type=str, default='_results/hyperband/hyperband.resample_5/weights.720')
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


actions = pd.read_csv('_results/hyperband/hyperband.resample_5/test.actions', sep='\t', header=None)
actions.columns = ['mode', 'epoch', 'score'] + list(range(actions.shape[1] - 4)) + ['aid']
actions = actions[actions.epoch == 720]

paths = np.array(actions[list(range(12))].drop_duplicates())

args = parse_args()
set_seeds(args.seed)

dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pin_memory=True, shuffle_test=False)

worker = CellWorker(num_nodes=args.num_nodes).cuda().eval()
# worker.load(args.pretrained_path)
# >>
state_dict = torch.load(args.pretrained_path)
all_keys = list(state_dict.keys())
for k in all_keys:
    if 'active_layer' in k:
        del state_dict[k]

worker.load_state_dict(state_dict)
# <<


child = Child(worker=worker, dataloaders=dataloaders)

# >>
worker.init_optimizer(
    opt=torch.optim.SGD,
    params=filter(lambda x: x.requires_grad, worker.parameters()),
    lr=0.0,
    momentum=0.9,
    weight_decay=5e-4
)

child.train_paths(paths)
# Fix messed up batchnorm

worker.layers
all_keys

for k1, v1 in worker.layers.named_children():
    for k2, v2 in v1.named_children():
        for k3, v3 in v2.named_children():
            for k4, v4 in v3.named_children():
                if k4 == 'bn':
                    print(list(v4.named_children()))


# <<

rewards = child.eval_paths(paths, mode='test', n=10)

res = {}

for path in tqdm(paths):
    worker.verbose = False
    worker.set_path(path)
    preds, labs = worker.predict(dataloaders=dataloaders, mode='test')
    res[''.join(map(str, path))] = {
        "preds" : to_numpy(preds),
        "labs"  : to_numpy(labs).squeeze(),
    }


labs = res[list(res.keys())[0]]['labs']
preds = np.array([res[k]['preds'] for k in res.keys()])
ppreds = np.exp(preds) / np.exp(preds.sum(-1, keepdims=True))

_ = plt.hist((preds.argmax(axis=-1) == labs).mean(axis=-1), 100)
_ = plt.axvline((preds.mean(axis=0).argmax(axis=-1) == labs).mean(), c='red')
show_plot()

