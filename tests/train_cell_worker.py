#!/usr/bin/env python

"""
    main.py
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from data import make_cifar_dataloaders
from workers import CellWorker

import basenet
from basenet.lr import LRSchedule

np.set_printoptions(linewidth=120)


# --
# Initialize child


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--train-size', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--reduce-p-survive', action="store_true")
    
    parser.add_argument('--architecture', type=str, default='00210102')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    basenet.helpers.set_seeds(args.seed)
    
    # --
    # IO
    print('train_cell_worker: make_cifar_dataloaders', file=sys.stderr)
    
    dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed)
    
    # --
    # Model
    print('train_cell_worker: CellWorker', file=sys.stderr)
    
    worker = CellWorker().cuda()
    
    lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)
    worker.init_optimizer(
        opt=torch.optim.SGD,
        params=worker.parameters(),
        lr_scheduler=lr_scheduler,
        momentum=0.9,
        weight_decay=5e-4
    )
    # print('worker ->', worker, file=sys.stderr)
    
    architecture = np.array(list(map(int, list(args.architecture))))
    assert len(architecture) == (4 * worker.num_nodes), "len(architecture) != 4 * worker.num_nodes"
    print('train_cell_worker: worker.set_path(%s)' % args.architecture, file=sys.stderr)
    worker.set_path(architecture)
    
    # --
    # Run
    
    print('train_cell_worker: run', file=sys.stderr)
    
    history = []
    for epoch in range(args.epochs):
        train_acc = worker.train_epoch(dataloaders)['acc']
        history.append({
            "epoch"     : int(epoch),
            "train_acc" : float(train_acc),
            "val_acc"   : float(worker.eval_epoch(dataloaders, mode='val')['acc']),
            "test_acc"  : float(worker.eval_epoch(dataloaders, mode='test')['acc']),
        })
        print(history[-1], file=sys.stderr)
        print(json.dumps(history[-1]))
        
    worker.save(args.outpath)