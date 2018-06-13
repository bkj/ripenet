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
from time import time
from datetime import datetime
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True

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
    parser.add_argument('--lr-max', type=float, default=0.1)
    
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--pretrained-path', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    mid = '%s_%s' % (args.architecture, datetime.now().strftime('%s'))
    
    print('train.py: architecture=%s' % args.architecture, file=sys.stderr)
    architecture = list(re.sub('[^0-9]','', args.architecture))
    architecture = np.array(list(map(int, architecture)))
    assert len(architecture) % 3 == 0, "len(archictecture) % 3 != 0"
    num_nodes = len(architecture) // 4
    
    # --
    # IO
    
    try:
        dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pin_memory=True)
    except:
        dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=True, seed=args.seed, pin_memory=True)
    
    # --
    # Model
    
    worker = CellWorker(num_nodes=num_nodes).to(torch.device('cuda'))
    
    lr_scheduler = getattr(HPSchedule, args.lr_schedule)(hp_max=args.lr_max, epochs=args.epochs)
    worker.init_optimizer(
        opt=torch.optim.SGD,
        params=filter(lambda x: x.requires_grad, worker.parameters()),
        hp_scheduler={"lr" : lr_scheduler},
        momentum=0.9,
        weight_decay=5e-4
    )
    
    worker.set_path(architecture)
    worker.trim_pipes()
    
    # --
    # Log
    
    config = vars(args)
    config['_pipes'] = worker.get_pipes()[0]
    json.dump(config, open(os.path.join(args.outpath, '%s.config' % mid), 'w'))
    
    # --
    # Run
    
    print('train_cell_worker: run', file=sys.stderr)
    
    logfile = open(os.path.join(args.outpath, '%s.log' % mid), 'w')
    rich_logfile = open(os.path.join(args.outpath, '%s.rich_log' % mid), 'w')
    
    t = time()
    for epoch in range(args.epochs):
        print('epoch=%d' % epoch, file=sys.stderr)
        train = worker.train_epoch(dataloaders, mode='train')
        val   = worker.eval_epoch(dataloaders, mode='val') if dataloaders['val'] else None
        test  = worker.eval_epoch(dataloaders, mode='test')
        
        elapsed_time = time() - t
        
        print(json.dumps(OrderedDict([
            ("epoch",        epoch),
            ("train_acc",    train['acc']),
            ("val_acc",      val['acc'] if val is not None else None),
            ("test_acc",     test['acc']),
            ("lr",           worker.hp['lr']),
            ("elapsed_time", elapsed_time),
        ])), file=logfile)
        logfile.flush()
        
        print(json.dumps(OrderedDict([
            ("epoch",        epoch),
            ("train",        train),
            ("val",          val if val is not None else None),
            ("test",         test),
            ("elapsed_time", elapsed_time),
        ])), file=rich_logfile)
        rich_logfile.flush()
    
    logfile.close()
    rich_logfile.close()
    
    worker.save(os.path.join(args.outpath, "%s.weights" % mid))

