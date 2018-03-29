#!/usr/bin/env python

"""
    main.py
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

from controllers import MicroLSTMController, HyperbandController
from children import LazyChild, Child
from workers import CellWorker
from data import make_cifar_dataloaders, make_mnist_dataloaders
from logger import HyperbandLogger

from basenet.helpers import to_numpy, set_seeds
from basenet.lr import LRSchedule

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion_mnist', 'mnist'])
    
    parser.add_argument('--child', type=str, default='child', choices=['lazy_child', 'child'])
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['reinforce', 'ppo', 'hyperband'])
    
    parser.add_argument('--epochs', type=int, default=20)  #
    parser.add_argument('--child-train-paths-per-epoch', type=int, default=352)     # Number of paths to use to train child network each epoch
    parser.add_argument('--controller-train-steps-per-epoch', type=int, default=4)  # Number of times to call RL step on controller per epoch
    parser.add_argument('--controller-train-paths-per-step', type=int, default=50)  # Number of paths to use to train controller per step
    parser.add_argument('--controller-eval-paths-per-epoch', type=int, default=352) # Number of paths to sample to quantify performance
    
    parser.add_argument('--controller-train-interval', type=int, default=1)         # Frequency of controller steps (in epochs)
    parser.add_argument('--controller-eval-interval', type=int, default=1)          # Frequency of running on test set (in epochs)
    parser.add_argument('--controller-train-mult', type=int, default=1)             # Increase train interval over time?
    
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
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

# >>
# Worker

sys.path.append('/home/bjohnson/ftune/')
from layers import AdaptiveMultiPool2d, Flatten

sys.path.append('/home/bjohnson/ripenet/workers/')
from cell_worker import _CellWorker, CellBlock

def BoltWorker(_CellWorker):
    
    def __init__(self, num_features, num_classes=200, num_nodes=2, num_branches=2):
        
        self.layers = nn.Sequential(*[
            CellBlock(in_channels=num_features, out_channels=num_features, num_nodes=num_nodes, num_branches=num_branches),
        ])
        
        self.classifier = nn.Sequential(*[
            AdaptiveMultiPool2d(output_size=(1, 1)),
            Flatten(),
            nn.Linear(in_features=num_features, out_features=num_classes),
            nn.LogSoftmax(),
        ])
    
    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x

# <<

state_dim = 32

if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    json.dump(vars(args), open(os.path.join(args.outpath, 'config'), 'w'))
    
    # --
    # IO
    
    # dataloaders = ...
    
    # --
    # Controller
    
    controller_args = {
        "output_length" : args.num_nodes,
        "output_channels" : args.num_ops,
        # Hyperband parameters
        "population_size" : args.population_size,
    }
    
    controller = HyperbandController(**controller_args)
    print("controller.population ->\n", controller.population, file=sys.stderr)
    
    # --
    # Worker
    
    worker = BoltWorker(num_nodes=args.num_nodes).cuda()
        
    if args.pretrained_path is not None:
        print('main.py: loading pretrained model %s' % args.pretrained_path, file=sys.stderr)
        worker.load_state_dict(torch.load(args.pretrained_path))
    
    # Save model on exit
    def save(suffix='final'):
        worker.save(os.path.join(args.outpath, 'weights.' + suffix))
    
    atexit.register(save)
    
    # --
    # Child
    
    print('main.py: child -> %s' % args.child, file=sys.stderr)
    
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
        
    # --
    # Run
    
    total_controller_steps = 0
    train_rewards, rewards = None, None
    controller_train_interval = args.controller_train_interval

    logger = HyperbandLogger(args.outpath)
    
    for epoch in range(args.epochs):
        print(('epoch=%d ' % epoch) + ('-' * 50), file=sys.stderr)
        
        # --
        # Train child
        
        states = Variable(torch.randn(args.child_train_paths_per_epoch, state_dim))
        train_actions, _, _ = controller(states)
        train_rewards = child.train_paths(train_actions)
        logger.log(epoch=epoch, rewards=train_rewards, actions=train_actions, mode='train')
        
        # --
        # Train controller
        
        if not (epoch + 1) % controller_train_interval:
            if args.hyperband_halving:
                total_controller_steps += 1
                
                save(suffix=str(epoch + 1)) # Checkpoint model
                
                rewards = child.eval_paths(controller.population, mode='val', n=10)
                logger.log(epoch=epoch, rewards=rewards, actions=controller.population, mode='val')
                
                controller_update = controller.hyperband_step(rewards, resample=args.hyperband_resample)
                
                # Update controller train interval
                controller_train_interval = sum([args.controller_train_interval * (args.controller_train_mult ** i) for i in range(total_controller_steps + 1)])
                logger.controller_log(epoch=epoch, controller_update=controller_update)
        
        # --
        # Eval best architecture on test set
        
        if not (epoch + 1) % args.controller_eval_interval:
            rewards = child.eval_paths(controller.population, mode='test', n=10)
            logger.log(epoch=epoch, rewards=rewards, actions=controller.population, mode='test')
    
    logger.close()