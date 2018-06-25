#!/usr/bin/env python

"""
    main.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from rsub import *
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from controllers import HyperbandController
from workers import CellWorker
from data import make_cifar_dataloaders, make_mnist_dataloaders
from logger import Logger

from basenet.helpers import to_numpy, set_seeds
from basenet.hp_schedule import HPSchedule, HPFind

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ops', type=int, default=6)   # Number of ops to sample
    parser.add_argument('--num-nodes', type=int, default=2) # Number of cells to sample
    parser.add_argument('--population-size', type=int, default=32)
    parser.add_argument('--train-size', type=float, default=0.9)     # Proportion of training data to use for training (vs validation) 
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


# --
# Run

args = parse_args()
set_seeds(args.seed)

dataloaders = make_cifar_dataloaders(
    train_size=args.train_size, download=False, seed=args.seed, pin_memory=True)

controller = HyperbandController(
    output_length=args.num_nodes,
    output_channels=args.num_ops,
    population_size=args.population_size,
)

worker = CellWorker(num_nodes=args.num_nodes).to(torch.device('cuda'))
worker.verbose = True

fitists = {}
for i, path in enumerate(controller.population):
    print(i, path)
    worker.set_path(path)
    
    hp_hist, loss_hist = HPFind.find(worker, dataloaders, mode='train', hp_init=1e-4)
    
    fitists[tuple(path)] = {
        "lrs"  : hp_hist.squeeze(),
        "loss" : loss_hist,
    }
    
    for p, v in fitists.items():
        lrs, loss = v['lrs'], v['loss']
        _ = plt.plot(np.log10(lrs), np.log10(loss), label=p)
        
    show_plot()

argmins = [pd.Series(v['loss']).rolling(10).mean().argmin() for v in fitists.values()]
best  = np.array([v['lrs'][a] for v,a in zip(fitists.values(), argmins)])

np.array(list(fitists.keys()))[np.argsort(best)]

