#!/usr/bin/env python

"""
    main.py
"""

import os
import re
import sys
import json
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from data import make_cifar_dataloaders
from workers import CellWorker

import basenet
from basenet.hp_schedule import HPSchedule
from basenet.helpers import to_numpy, set_seeds

np.set_printoptions(linewidth=120)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default='delete-me')
    parser.add_argument('--architecture', type=str, default='0002_0112')
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr-schedule', type=str, default='linear_cycle')
    parser.add_argument('--lr-init', type=float, default=0.1)
    
    parser.add_argument('--output-length', type=int, default=2)   # Number of ops to sample
    
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--pretrained-path', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    try:
        dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pin_memory=True)
    except:
        dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=True, seed=args.seed, pin_memory=True)
    
    # --
    # Model
    
    worker = CellWorker(num_nodes=args.output_length).cuda()
    worker.verbose = True
    
    # if args.pretrained_path is not None:
    #     print('main.py: loading pretrained model %s' % args.pretrained_path, file=sys.stderr)
    #     worker.load_state_dict(torch.load(args.pretrained_path))
    
    # --
    # Training options
    
    lr_scheduler = getattr(HPSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)
    worker.init_optimizer(
        opt=torch.optim.SGD,
        params=filter(lambda x: x.requires_grad, worker.parameters()),
        hp_scheduler={"lr" : lr_scheduler},
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # --
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
    json.dump(config, open('%s.%s.config' % (args.outpath, args.architecture), 'w'))
    
    # --
    # Run
    
    print('train_cell_worker: run', file=sys.stderr)
    
    logfile = open(args.outpath + '.log', 'w')
    
    for epoch in range(args.epochs):
        print('epoch=%d' % epoch, file=sys.stderr)
        train_acc = worker.train_epoch(dataloaders, mode='train')['acc']
        val_acc   = worker.eval_epoch(dataloaders, mode='val')['acc'] if dataloaders['val'] else -1
        test_acc  = worker.eval_epoch(dataloaders, mode='test')['acc']
        print(json.dumps(OrderedDict([
            ("epoch",     int(epoch)),
            ("train_acc", float(train_acc)),
            ("val_acc",   float(val_acc)),
            ("test_acc",  float(test_acc)),
        ])), file=logfile)
        logfile.flush()
    
    logfile.close()
    
    worker.save("%s.%s.weights" % (args.outpath, args.architecture))

