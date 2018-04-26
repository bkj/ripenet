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
from glob import glob

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
    # parser.add_argument('--config-path', type=str, default='runs/run_2nodes/results/00350162_1524646450.config')
    # parser.add_argument('--weight-path', type=str, default='runs/run_2nodes/results/00350162_1524646450.weights')
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def load_worker(config_path, weight_path):
    
    config = json.load(open(config_path))
    architecture = list(re.sub('[^0-9]','', config['architecture']))
    architecture = np.array(list(map(int, architecture)))
    num_nodes = len(architecture) // 4
    
    worker = CellWorker(num_nodes=num_nodes).cuda()
    worker.init_optimizer(opt=torch.optim.SGD, params=worker.parameters(), lr=0.0)
    worker.set_path(architecture)
    worker.trim_pipes()
    
    worker.load(weight_path)
    
    worker.verbose = True
    
    return worker


if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pin_memory=True, 
        shuffle_train=False, shuffle_test=False)
    
    # --
    # Load + Predict
    
    architectures = set([os.path.basename(f).split('.')[0] for f in glob('./runs/run_2nodes/results/*')])
    
    preds = {
        "train" : {},
        "val"   : {},
        "test"  : {},
    }
    
    for architecture in architectures:
        config_path = os.path.join('./runs/run_2nodes/results/', '%s.config' % architecture)
        weight_path = os.path.join('./runs/run_2nodes/results/', '%s.weights' % architecture)
        
        if os.path.exists(config_path) and os.path.exists(weight_path):
            worker = load_worker(config_path, weight_path)
            
            train_preds, train_targets = worker.predict(dataloaders, mode='train')
            train_acc = (train_preds.max(dim=-1)[1] == train_targets).float().mean()
            print('%s -> train_acc=%f' % (architecture, train_acc))
            print(to_numpy(train_targets)[:10])
            preds['train'][architecture] = {
                "preds"   : to_numpy(train_preds),
                "targets" : to_numpy(train_targets),
            }
            
            test_preds, test_targets = worker.predict(dataloaders, mode='test')
            test_acc = (test_preds.max(dim=-1)[1] == test_targets).float().mean()
            print('%s -> test_acc=%f' % (architecture, test_acc))
            print(to_numpy(test_targets)[:10])
            preds['test'][architecture] = {
                "preds"   : to_numpy(test_preds),
                "targets" : to_numpy(test_targets),
            }
            
        else:
            print('config or weight does not exist -> %s' % architecture, file=sys.stderr)


train_targets = list(preds['train'].values())[0]['targets']
for v in preds['train'].values():
    assert (v['targets'] == train_targets).all()

test_targets = list(preds['test'].values())[0]['targets']
for v in preds['test'].values():
    assert (v['targets'] == test_targets).all()


all_train_preds = np.array([v['preds'] for v in preds['train'].values()])
np.save('train_preds', all_train_preds)
np.save('train_targets', train_targets)

all_test_preds = np.array([v['preds'] for v in preds['test'].values()])
np.save('test_preds', all_test_preds)
np.save('test_targets', test_targets)

# --
# Does ensembling these models work?

targets = list(preds['test'].values())[0]['targets']
for v in preds['test'].values():
    assert (v['targets'] == targets).all()


all_preds = np.array([v['preds'] for v in preds['test'].values()])
all_preds = all_preds[np.isnan(preds).sum(axis=-1).sum(axis=-1) == 0]

ind_accs  = (all_preds.argmax(axis=-1) == targets).mean(axis=-1)
o = np.argsort(ind_accs)[::-1]
all_preds, ind_accs = all_preds[o], ind_accs[o]

ens_accs = [(all_preds[:i].mean(axis=0).argmax(axis=-1) == targets).mean(axis=-1) for i in range(1, len(all_preds) + 1)]

_ = plt.plot(ind_accs[:50], 'ro', alpha=0.25)
_ = plt.plot(ens_accs[:50], 'bo', alpha=0.25)
show_plot()

# Yes

