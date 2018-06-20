#!/usr/bin/env python

"""
    main.py
"""

from __future__ import division, print_function

import os
import sys
import h5py
import json
import argparse
import numpy as np
from glob import glob
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from controllers import HyperbandController
from children import Child
from workers import BoltWorker
from logger import Logger

from basenet.helpers import to_numpy, set_seeds
from basenet.hp_schedule import HPSchedule

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    
    parser.add_argument('--epochs', type=int, default=1000)  #
    parser.add_argument('--child-train-paths-per-epoch', type=int, default=32) # Number of paths to use to train child network each epoch

    parser.add_argument('--controller-train-paths-per-step',   type=int, default=50)   # Number of paths to use to train controller per step
    parser.add_argument('--controller-eval-paths-per-epoch',   type=int, default=32)   # Number of paths to sample to quantify performance
    parser.add_argument('--controller-predict-paths-per-step', type=int, default=128)  # Number of paths to sample to quantify performance
    parser.add_argument('--controller-train-interval',   type=int, default=5)          # Frequency of controller steps (in epochs)
    parser.add_argument('--controller-predict-interval', type=int, default=20)         # Frequency of running on test set (in epochs)
    
    parser.add_argument('--num-ops', type=int, default=8)          # Number of ops to sample
    parser.add_argument('--num-nodes', type=int, default=2)        # Number of cells to sample
    parser.add_argument('--population-size', type=int, default=32) # Number of architectures to sample
    
    parser.add_argument('--child-lr-init', type=float, default=0.1)
    parser.add_argument('--child-lr-schedule', type=str, default='sgdr')
    parser.add_argument('--child-lr-epochs', type=int, default=1000) # For LR schedule
    parser.add_argument('--child-sgdr-period-length', type=float, default=5)
    parser.add_argument('--child-sgdr-t-mult',  type=float, default=1)
    
    parser.add_argument('--seed', type=int, default=456)
    return parser.parse_args()

# >>
# Worker

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.length = len(h5py.File(h5_path, 'r'))
    
    def __getitem__(self, index):
        h5_file = h5py.File(self.h5_path, 'r')
        
        record = h5_file[str(index)]
        res = (
            torch.from_numpy(record['data'].value),
            record['target'].value,
        )
        
        h5_file.close()
        return res
        
    def __len__(self):
        return self.length

def get_precomputed_loaders(cache='data/cub_precomputed', batch_size=128, shuffle=True, num_workers=8, **kwargs):
    
    kwargs.update({
        "batch_size"  : batch_size,
        "shuffle"     : shuffle,
        "num_workers" : num_workers,
    })
    
    loaders = {}
    for cache_path in glob(os.path.join(cache, '*.h5')):
        print("get_precomputed_loaders: loading cache %s" % cache_path, file=sys.stderr)
        cache_name = os.path.basename(cache_path).split('.')[0]
        loaders[cache_name] = torch.utils.data.DataLoader(H5Dataset(cache_path), **kwargs)
    
    return loaders

# # >>
# # Test/val split

# import h5py
# from sklearn.model_selection import train_test_split

# f = h5py.File('data/cub_precomputed/val.h5')

# lookup = [(k, v['target'].value) for k,v in f.items()]
# idx, lab = list(zip(*lookup))

# train_idx, test_idx = train_test_split(idx, train_size=0.5, stratify=lab, random_state=123)

# val_test = h5py.File('data/cub_precomputed/val_test.h5')
# for i, idx in enumerate(train_idx):
#     val_test['%s/data' % i] = f[idx]['data'].value
#     val_test['%s/target' % i] = f[idx]['target'].value

# test_test = h5py.File('data/cub_precomputed/test_test.h5')
# for i, idx in enumerate(test_idx):
#     test_test['%s/data' % i] = f[idx]['data'].value
#     test_test['%s/target' % i] = f[idx]['target'].value

# val_test.flush()
# val_test.close()
# test_test.flush()
# test_test.close()

# # <<

state_dim = 32

if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    json.dump(vars(args), open(os.path.join(args.outpath, 'config'), 'w'))
    
    def save(worker, suffix='final'):
        worker.save(os.path.join(args.outpath, 'weights.' + suffix))
    
    # --
    # Data
    
    print('get_precomputed_loaders', file=sys.stderr)
    dataloaders = get_precomputed_loaders()
    dataloaders = {
        "train"  : dataloaders['train_fixed'],
        "val"    : dataloaders['val_test'],
        "test"   : dataloaders['test_test'],
    }
    print('dataloaders.keys() ->', list(dataloaders.keys()), file=sys.stderr)
    
    # --
    # Worker
    
    def make_worker():
        worker = BoltWorker(num_nodes=args.num_nodes).to(torch.device('cuda'))
        
        # Save model on exit
        
        lr_scheduler_kwargs = {
            "hp_init"       : args.child_lr_init,
            "epochs"        : args.child_lr_epochs,
            "period_length" : args.child_sgdr_period_length,
            "t_mult"        : args.child_sgdr_t_mult,
        }
        if args.child_lr_schedule == 'sgdr':
            del lr_scheduler_kwargs['epochs']
        
        lr_scheduler = getattr(HPSchedule, args.child_lr_schedule)(**lr_scheduler_kwargs)
        
        worker.init_optimizer(
            opt=torch.optim.SGD,
            params=[p for p in worker.parameters() if p.requires_grad],
            hp_scheduler={
                "lr" : lr_scheduler,
            },
            momentum=0.9,
            weight_decay=5e-4,
            clip_grad_norm=1e3,
        )
        
        return worker
    
    # --
    # Logger
    
    child = Child(worker=make_worker(), dataloaders=dataloaders)
    controller = HyperbandController(
        output_length   = args.num_nodes,
        output_channels = args.num_ops,
        population_size = args.population_size,
    )
    logger = Logger(args.outpath)
    
    # --
    # Run
    
    n_hivar = 1
    n_lovar = 8
    for epoch in range(args.epochs):
        print(('epoch=%d ' % epoch) + ('-' * 50), file=sys.stderr)
        
        # Train child
        train_actions = controller(n_paths=args.child_train_paths_per_epoch)
        train_rewards = child.train_paths(train_actions)
        logger.log(epoch=epoch, rewards=train_rewards, actions=train_actions, mode='train')
        
        # Train controller
        if not (epoch + 1) % args.controller_train_interval:
            # Checkpoint model
            save(child.worker, suffix='most_recent')
            
            # Log test loss w/ lower variance
            actions = controller.population
            rewards = child.eval_paths(actions, mode='test', n=n_lovar)
            logger.log(epoch=epoch, rewards=rewards, actions=actions, mode='test', extra={"n" : n_lovar})
            
            # Compute eval loss w/ lower variance
            actions = controller.population
            rewards = child.eval_paths(actions, mode='val', n=n_lovar)
            logger.log(epoch=epoch, rewards=rewards, actions=actions, mode='val', extra={"n" : n_lovar})
            
            # Then take controller step
            controller_update = controller.hyperband_step(actions=actions, rewards=rewards, resample=True)
            logger.controller_log(epoch=epoch, controller_update=controller_update)
            
            # >>
            # Reset worker each time a step is taken
            print('cub.py: resetting worker', file=sys.stderr)
            child.worker = make_worker()
            # <<
        else:
            # Log test loss w/ high variance
            actions = controller.population
            rewards = child.eval_paths(actions, mode='test', n=n_hivar)
            logger.log(epoch=epoch, rewards=rewards, actions=actions, mode='test', extra={"n" : n_hivar})
            
        # # Take top-k models on validation set, log performance on test set
        # if (not (epoch + 1) % args.controller_predict_interval):
        #     val_n  = 32
        #     test_n = 32
        #     topk   = 5
            
        #     actions  = controller.population
        #     topk_idx = child.eval_paths(actions, mode='val', n=val_n).squeeze().topk(topk)[1]
        #     rewards  = child.eval_paths(actions[topk_idx], mode='test', n=test_n)
            
        #     logger.log(epoch=epoch, rewards=rewards, actions=actions[topk_idx], mode='test', extra={
        #         "predict"      : True,
        #         "topk"         : topk,
        #         "test_n"       : test_n,
        #         "val_n"        : val_n, 
        #         "action_order" : [str(action) for action in to_numpy(actions[topk_idx])]
        #     })
    
    logger.close()