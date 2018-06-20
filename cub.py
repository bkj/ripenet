#!/usr/bin/env python

"""
    main.py
"""

import os
import sys
import h5py
import json
import atexit
import argparse
import numpy as np
from glob import glob
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from controllers import MicroLSTMController, HyperbandController
from children import LazyChild, Child
from workers import CellWorker, BoltWorker
from data import make_cifar_dataloaders, make_mnist_dataloaders
from logger import Logger

from basenet.helpers import to_numpy, set_seeds
from basenet.hp_schedule import HPSchedule

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['reinforce', 'ppo', 'hyperband'])
    
    parser.add_argument('--epochs', type=int, default=20)  #
    parser.add_argument('--child-train-paths-per-epoch', type=int, default=200)      # Number of paths to use to train child network each epoch
    # parser.add_argument('--controller-train-steps-per-epoch', type=int, default=4) # Number of times to call RL step on controller per epoch
    
    parser.add_argument('--controller-train-paths-per-step',   type=int, default=50)   # Number of paths to use to train controller per step
    parser.add_argument('--controller-eval-paths-per-epoch',   type=int, default=32)   # Number of paths to sample to quantify performance
    parser.add_argument('--controller-predict-paths-per-step', type=int, default=128)   # Number of paths to sample to quantify performance
    
    parser.add_argument('--controller-train-interval',   type=int, default=1)         # Frequency of controller steps (in epochs)
    parser.add_argument('--controller-eval-interval',    type=int, default=1)         # Frequency of running on test set (in epochs)
    parser.add_argument('--controller-predict-interval', type=int, default=20)        # Frequency of running on test set (in epochs)
    
    parser.add_argument('--controller-train-mult',       type=int, default=1)         # Increase train interval over time?
    
    parser.add_argument('--num-ops', type=int, default=6)   # Number of ops to sample
    parser.add_argument('--num-nodes', type=int, default=2) # Number of cells to sample
    
    # RL Parameters
    parser.add_argument('--temperature', type=float, default=1)       # Temperature for logit -- higher means more entropy 
    parser.add_argument('--clip-logits', type=float, default=-1)      # Clip logits
    parser.add_argument('--entropy-penalty', type=float, default=0.0) # Penalize entropy 
    parser.add_argument('--controller-lr', type=float, default=0.001)
    
    # Hyperband parameters
    parser.add_argument('--population-size', type=int, default=8)
    parser.add_argument('--hyperband-halving', action="store_true")
    parser.add_argument('--hyperband-resample', action="store_true")
    
    parser.add_argument('--child-lr-init', type=float, default=0.1)
    parser.add_argument('--child-lr-schedule', type=str, default='constant')
    parser.add_argument('--child-lr-epochs', type=int, default=1000) # For LR schedule
    parser.add_argument('--child-sgdr-period-length', type=float, default=10)
    parser.add_argument('--child-sgdr-t-mult',  type=float, default=2)
    
    parser.add_argument('--train-size', type=float, default=0.9)     # Proportion of training data to use for training (vs validation) 
    parser.add_argument('--pretrained-path', type=str, default=None)
    
    parser.add_argument('--reset-model-interval', type=int, default=-1)
    
    parser.add_argument('--seed', type=int, default=123)
    
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
    
    # --
    # IO
    
    print('get_precomputed_loaders', file=sys.stderr)
    dataloaders = get_precomputed_loaders()
    dataloaders = {
        "train"  : dataloaders['train_fixed'],
        "val"    : dataloaders['val_test'],
        "test"   : dataloaders['test_test'],
    }
    print('dataloaders.keys() ->', list(dataloaders.keys()), file=sys.stderr)
    
    # --
    # Controller
    
    controller_args = {
        "output_length" : args.num_nodes,
        "output_channels" : args.num_ops,
        # RL parameters
        "input_dim" : state_dim,
        "temperature" : args.temperature,
        "clip_logits" : args.clip_logits,
        "opt_params" : {
            "lr" : args.controller_lr,
        },
        # Hyperband parameters
        "population_size" : args.population_size,
    }
    
    if args.algorithm == 'hyperband':
        controller = HyperbandController(**controller_args)
        print("controller.population ->\n", controller.population, file=sys.stderr)
    else:
        controller = MicroLSTMController(**controller_args)
    
    # --
    # Worker
    
    set_seeds(args.seed)
    worker = BoltWorker(num_nodes=args.num_nodes).to(torch.device('cuda'))
    
    if args.pretrained_path is not None:
        print('main.py: loading pretrained model %s' % args.pretrained_path, file=sys.stderr)
        worker.load_state_dict(torch.load(args.pretrained_path))
    
    # Save model on exit
    def save(suffix='final'):
        worker.save(os.path.join(args.outpath, 'weights.' + suffix))
    
    atexit.register(save)
    
    # --
    # Child
    
    lr_scheduler_kwargs = {
        "hp_init" : args.child_lr_init,
        "epochs" : args.child_lr_epochs,
        "period_length" : args.child_sgdr_period_length,
        "t_mult" : args.child_sgdr_t_mult,
    }
    if args.child_lr_schedule == 'sgdr':
        del lr_scheduler_kwargs['epochs']
    
    lr_scheduler = getattr(HPSchedule, args.child_lr_schedule)(**lr_scheduler_kwargs)
    worker.init_optimizer(
        opt=torch.optim.SGD,
        params=filter(lambda x: x.requires_grad, worker.parameters()),
        hp_scheduler={
            "lr" : lr_scheduler
        },
        momentum=0.9,
        weight_decay=5e-4,
        clip_grad_norm=1e3,
    )
    
    child = Child(worker=worker, dataloaders=dataloaders)
    
    # --
    # Run
    
    total_controller_steps = 0
    train_rewards, rewards = None, None
    controller_train_interval = args.controller_train_interval
    logger = Logger(args.outpath)
    
    for epoch in range(args.epochs):
        print(('epoch=%d ' % epoch) + ('-' * 50), file=sys.stderr)
        
        # Train child
        states = Variable(torch.randn(args.child_train_paths_per_epoch, state_dim))
        train_actions, _, _ = controller(states)
        train_rewards = child.train_paths(train_actions)
        logger.log(epoch=epoch, rewards=train_rewards, actions=train_actions, mode='train')
        
        
        # Log test loss w/ high variance
        def do_eval(n=1):
            if args.algorithm == 'hyperband':
                rewards = child.eval_paths(controller.population, mode='test', n=1)
                logger.log(epoch=epoch, rewards=rewards, actions=controller.population, mode='test', extra={"n" : 1})
            else:
                states = Variable(torch.randn(args.controller_eval_paths_per_epoch, state_dim))
                actions, _, _ = controller(states)
                rewards = child.eval_paths(actions, mode='test', n=1)
                logger.log(epoch=epoch, rewards=rewards, actions=actions, mode='test', extra={
                    "n" : 1
                })
        
        if (not (epoch + 1) % args.controller_eval_interval) and ((epoch + 1) % controller_train_interval):
            do_eval(n=1)
        
        # Train controller
        if not (epoch + 1) % controller_train_interval:
            total_controller_steps += 1
            
            # Change frequency of controller updates (maybe)
            # controller_train_interval = sum([args.controller_train_interval * (args.controller_train_mult ** i) 
            #     for i in range(total_controller_steps + 1)])
            
            # Log test loss w/ low(er) variance
            do_eval(n=10)
            
            # Checkpoint model
            save(suffix='most_recent') 
            
            # Controller step
            if (args.algorithm == 'hyperband') and (args.hyperband_halving):
                rewards = child.eval_paths(controller.population, mode='val', n=10)
                
                controller_update = controller.hyperband_step(rewards, resample=args.hyperband_resample)
                
                logger.log(epoch=epoch, rewards=rewards, actions=controller.population, mode='val')
                logger.controller_log(epoch=epoch, controller_update=controller_update)
                
            else:
                states = Variable(torch.randn(args.controller_train_paths_per_step, state_dim))
                actions, log_probs, entropies = controller(states)
                rewards = child.eval_paths(actions, mode='val', n=10)
                
                step_results = controller.reinforce_step(rewards, log_probs=log_probs, entropies=entropies, entropy_penalty=args.entropy_penalty)
                
                logger.log(epoch=epoch, rewards=rewards, actions=actions, mode='val', extra=step_results)
        
        # Take top-k models on validation set, log performance on test set
        if (not (epoch + 1) % args.controller_predict_interval):
            val_n, test_n, topk = 16, 32, 5
            if args.algorithm != 'hyperband':
                states = Variable(torch.randn(args.controller_predict_paths_per_step, state_dim))
                actions, _, _ = controller(states)
            else:
                actions = controller.population
            
            # apply topk on val to test
            topk_idx = child.eval_paths(actions, mode='val', n=val_n).squeeze().topk(topk)[1]
            rewards  = child.eval_paths(actions[topk_idx], mode='test', n=test_n)
            
            logger.log(epoch=epoch, rewards=rewards, actions=actions[topk_idx], mode='test', extra={
                "predict"      : True,
                "topk"         : topk,
                "test_n"       : test_n,
                "val_n"        : val_n, 
                "action_order" : [str(action) for action in to_numpy(actions[topk_idx])]
            })
    
    logger.close()