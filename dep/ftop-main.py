#!/usr/bin/env python

"""
    ftop-main.py
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

from controllers import FTopLSTMController
from children import LazyChild, Child
from workers import FTopWorker
from data import make_cifar_dataloaders, make_mnist_dataloaders
from logger import Logger

from basenet.helpers import to_numpy, set_seeds

np.set_printoptions(linewidth=120)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion_mnist', 'mnist'])
    
    parser.add_argument('--child', type=str, default='lazy_child', choices=['lazy_child', 'child'])
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['reinforce', 'ppo'])
    
    parser.add_argument('--epochs', type=int, default=20)  #
    parser.add_argument('--child-train-paths-per-epoch', type=int, default=400)     # Number of paths to use to train child network each epoch
    parser.add_argument('--controller-train-steps-per-epoch', type=int, default=2)  # Number of times to call RL step on controller per epoch
    parser.add_argument('--controller-train-paths-per-step', type=int, default=80)  # Number of paths to use to train controller per step
    parser.add_argument('--controller-eval-paths-per-epoch', type=int, default=200) # Number of paths to sample to quantify performance
    parser.add_argument('--controller-eval-interval', type=int, default=5)          # Number of paths to sample to quantify performance
    
    parser.add_argument('--test-topk', type=int, default=-1) # Number of paths to sample to quantify performance
    
    parser.add_argument('--num-ops', type=int, default=6) # Number of ops to sample
    parser.add_argument('--num-nodes', type=int, default=2) # Number of cells to sample
    
    parser.add_argument('--temperature', type=float, default=1) # Temperature for logit -- higher means more entropy 
    parser.add_argument('--entropy-penalty', type=float, default=0.0) # Penalize entropy 
    
    parser.add_argument('--controller-lr', type=float, default=0.001)
    parser.add_argument('--child-lr-init', type=float, default=0.1)
    
    parser.add_argument('--train-size', type=float, default=0.9)     # Proportion of training data to use for training (vs validation) 
    parser.add_argument('--pretrained-path', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


state_dim = 32

if __name__ == "__main__":
    
    args = parse_args()
    set_seeds(args.seed)
    
    json.dump(vars(args), open(args.outpath + '.config', 'w'))

    # --
    # Controller

    controller_kwargs = {
        "input_dim" : state_dim,
        "output_length" : args.num_nodes * 2,
        "output_channels" : args.num_ops,
        "temperature" : args.temperature,
        "opt_params" : {
            "lr" : args.controller_lr,
        }
    }

    controller = FTopLSTMController(**controller_kwargs)

    # --
    # IO

    if args.dataset == 'cifar10':
        print('train_cell_worker: make_cifar_dataloaders', file=sys.stderr)
        dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed)
    elif 'mnist' in args.dataset:
        print('train_cell_worker: make_mnist_dataloaders (%s)' % args.dataset, file=sys.stderr)
        dataloaders = make_mnist_dataloaders(train_size=args.train_size, download=False, seed=args.seed, pretensor=True, mode=args.dataset)
    else:
        raise Exception()

    # --
    # Worker
    
    worker = FTopWorker(num_nodes=args.num_nodes, num_branches=2).cuda()
    
    if args.pretrained_path is not None:
        print('main.py: loading pretrained model %s' % args.pretrained_path, file=sys.stderr)
        worker.load_state_dict(torch.load(args.pretrained_path))
    
    # Save model on exit
    def save():
        worker.save(args.outpath + '.weights')
    
    atexit.register(save)
    
    # --
    # Child
    
    print('main.py: child -> %s' % args.child, file=sys.stderr)
    
    if args.child == 'lazy_child':
        child = LazyChild(worker=worker, dataloaders=dataloaders)
    elif args.child == 'child':
        worker.init_optimizer(
            opt=torch.optim.SGD,
            params=worker.parameters(),
            lr=args.child_lr_init,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        child = Child(worker=worker, dataloaders=dataloaders)
    else:
        raise Exception('main.py: unknown child %s' % args.child, file=sys.stderr)
        
    # --
    # Run
    
    total_controller_steps = 0
    logger = Logger(args.outpath)
    
    for epoch in range(args.epochs):
        print(('epoch=%d ' % epoch) + ('-' * 50), file=sys.stderr)
        
        # --
        # Train child
        
        if args.child != 'lazy_child':
            states = Variable(torch.randn(args.child_train_paths_per_epoch, state_dim))
            actions, _, _ = controller(states)
            child.train_paths(actions)
        
        # --
        # Train controller
        
        for controller_step in range(args.controller_train_steps_per_epoch):
            
            states = Variable(torch.randn(args.controller_train_paths_per_step, state_dim))
            actions, log_probs, entropies = controller(states)
            rewards = child.eval_paths(actions, n=1)
            
            if args.algorithm == 'reinforce':
                controller.reinforce_step(rewards, log_probs=log_probs, entropies=entropies, entropy_penalty=args.entropy_penalty)
            elif args.algorithm == 'ppo':
                controller.ppo_step(rewards, states=states, actions=actions, entropy_penalty=args.entropy_penalty)
            else:
                raise Exception('unknown algorithm %s' % args.algorithm, file=sys.stderr)
            
            total_controller_steps += 1
            logger.log(total_controller_steps, child, rewards, actions, mode='val')
            
        # --
        # Eval best architecture on test set
        
        if not (epoch + 1) % args.controller_eval_interval:
            states = Variable(torch.randn(args.controller_eval_paths_per_epoch, state_dim))
            actions, log_probs, entropies = controller(states)
            if args.test_topk > 0:
                N = 3
                topk_idx = child.eval_paths(actions, n=N, mode='val').squeeze().topk(args.test_topk)[1]
                rewards  = child.eval_paths(actions[topk_idx], mode='test')
            else:
                rewards = child.eval_paths(actions, mode='test')
                
            logger.log(total_controller_steps, child, rewards, actions, mode='test')
    
    logger.close()