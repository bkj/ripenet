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
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from children import Child
from workers import CellWorker
from data import make_cifar_dataloaders

from basenet.helpers import to_numpy, set_seeds
from basenet.lr import LRSchedule

np.set_printoptions(linewidth=120)

# --
# CLI


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
    parser.add_argument('--pretrained-path', type=str, default='_results/hyperband/hyperband.resample_4/weights.40')
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()




args = parse_args()
set_seeds(args.seed)

dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pin_memory=True)

worker = CellWorker(num_nodes=args.num_nodes).cuda()
worker.load(args.pretrained_path)

lr_scheduler = getattr(LRSchedule, args.child_lr_schedule)(
    lr_init=args.child_lr_init,
    epochs=args.child_lr_epochs,
    period_length=args.child_sgdr_period_length,
    t_mult=args.child_sgdr_t_mult,
)
worker.init_optimizer(
    opt=torch.optim.SGD,
    params=filter(lambda x: x.requires_grad, worker.parameters()),
    lr_scheduler=lr_scheduler,
    momentum=0.9,
    weight_decay=5e-4
)
    
child = Child(worker=worker, dataloaders=dataloaders)
rewards = child.eval_paths(paths, mode='test', n=10)
