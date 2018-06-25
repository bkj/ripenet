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
    parser.add_argument('--outpath', type=str, default='delete-me')
    parser.add_argument('--architecture', type=str, required="0000_0000")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--num-nodes', type=int, default=2) # Number of cells to sample
    parser.add_argument('--train-size', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pin_memory=True)
    worker = CellWorker(num_nodes=args.num_nodes).to(torch.device('cuda'))
    worker.verbose = True
    
    hp_hist, loss_hist = HPFind.find(worker, dataloaders, mode='train', hp_init=1e-4, smooth_loss=True)
    opt_lr = HPFind.get_optimal_hp(hp_hist, loss_hist)
    
    _ = plt.plot(np.log10(hp_hist), np.log10(loss_hist))
    show_plot()
    
    # --
    # Training options
    
    lr_scheduler = getattr(HPSchedule, args.lr_schedule)(hp_max=opt_lr, epochs=args.epochs)
    worker.init_optimizer(
        opt=torch.optim.SGD,
        params=filter(lambda x: x.requires_grad, worker.parameters()),
        hp_scheduler={
            "lr" : lr_scheduler
        },
        momentum=0.9,
        # weight_decay=5e-4
    )
    
    # Set architectecture
    architecture = list(re.sub('[^0-9]','', args.architecture))
    architecture = np.array(list(map(int, architecture)))
    assert len(architecture) == (4 * worker.num_nodes), "len(architecture) != 4 * worker.num_nodes"
    print('train_cell_worker: worker.set_path(%s)' % args.architecture, file=sys.stderr)
    worker.set_path(architecture)
    worker.trim_pipes()
    
    print(worker, file=sys.stderr)
    
    cell_pipes = worker.get_pipes()[0]
    config = vars(args)
    print('pipes ->', cell_pipes, file=sys.stderr)
    config['_pipes'] = cell_pipes
    json.dump(config, open(args.outpath + '.config', 'w'))
    
    # --
    # Run
    
    print('train_cell_worker: run', file=sys.stderr)
    
    logfile = open(args.outpath + '.log', 'w')
    
    history = []
    worker.verbose = True
    for epoch in range(args.epochs):
        print('epoch=%d' % epoch, file=sys.stderr)
        train_acc = worker.train_epoch(dataloaders)['acc']
        history.append(OrderedDict([
            ("epoch",     int(epoch)),
            ("train_acc", float(train_acc)),
            ("val_acc",   float(worker.eval_epoch(dataloaders, mode='val')['acc']) if dataloaders['val'] else None),
            ("test_acc",  float(worker.eval_epoch(dataloaders, mode='test')['acc'])),
        ]))
        print(json.dumps(history[-1]), file=logfile)
        logfile.flush()

    worker.save(args.outpath + '.weights')
    logfile.close()
