#!/usr/bin/env python

"""
    main.py
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from controllers import MicroLSTMController
from children import LazyChild, Child
from workers import CellWorker
from data import make_cifar_dataloaders, make_mnist_dataloaders

from basenet.helpers import to_numpy

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=str, default='lazy_child', choices=['lazy_child', 'child'])
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['reinforce', 'ppo'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion_mnist'])
    
    parser.add_argument('--epochs', type=int, default=20)  #
    parser.add_argument('--child-train-paths-per-epoch', type=int, default=200) # Number of paths to use to train child network each epoch
    parser.add_argument('--controller-train-steps-per-epoch', type=int, default=5)   # Number of times to call RL step on controller per epoch
    parser.add_argument('--controller-train-paths-per-step', type=int, default=40)  # Number of paths to use to train controller per step
    parser.add_argument('--controller-eval-paths-per-epoch',   type=int, default=100) # Number of paths to sample to quantify performance
    
    parser.add_argument('--num-ops', type=int, default=6)     # Number of ops to sample
    parser.add_argument('--num-nodes', type=int, default=2)     # Number of cells to sample
    
    parser.add_argument('--temperature', type=float, default=1.0)     # Temperature for logit -- higher means more entropy 
    parser.add_argument('--clip-logits', type=float, default=-1.0)
    
    parser.add_argument('--entropy-penalty', type=float, default=0.0)   # Penalize entropy 
    
    parser.add_argument('--controller-lr', type=float, default=0.001)
    
    parser.add_argument('--train-size', type=float, default=0.9)     # Proportion of training data to use for training (vs validation) 
    parser.add_argument('--pretrained-path', type=str, default=None)
    parser.add_argument('--outpath', type=str, default='last-run')
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

args = parse_args()

json.dump(vars(args), open(args.outpath + '.config', 'w'))

# --
# Parameters

state_dim = 32

# --
# Controller

controller_kwargs = {
    "input_dim" : state_dim,
    "output_length" : args.num_nodes,
    "output_channels" : args.num_ops,
    "temperature" : args.temperature,
    "clip_logits" : args.clip_logits,
    "opt_params" : {
        "lr" : args.controller_lr,
    }
}

controller = MicroLSTMController(**controller_kwargs)

# --
# Data

if args.dataset == 'cifar10':
    dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed)
elif args.dataset == 'fashion_mnist':
    dataloaders = make_mnist_dataloaders(train_size=args.train_size, download=False, seed=args.seed, mode='fashion_mnist', pretensor=True)
else:
    raise Exception()

# --
# Worker

if args.dataset == 'cifar10':
    worker = CellWorker(num_nodes=args.num_nodes).cuda()
elif args.dataset == 'fashion_mnist':
    worker = CellWorker(input_channels=1, num_blocks=[1, 1, 1], num_channels=[16, 32, 64], num_nodes=args.num_nodes).cuda()
else:
    raise Exception()

if args.pretrained_path is not None:
    print('main.py: loading pretrained model %s' % args.pretrained_path, file=sys.stderr)
    worker.load_state_dict(torch.load(args.pretrained_path))


# --
# Child

print('main.py: child -> %s' % args.child, file=sys.stderr)
if args.child == 'lazy_child':
    child = LazyChild(worker=worker, dataloaders=dataloaders)
elif args.child == 'child':
    worker.init_optimizer(
        opt=torch.optim.SGD,
        params=worker.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    child = Child(worker=worker, dataloaders=dataloaders)
else:
    raise Exception('main.py: unknown child %s' % args.child, file=sys.stderr)


# >>

# actions = torch.LongTensor([
#     [0, 0, 2, 1, 1, 0, 2, 0],
#     [0, 0, 2, 1, 1, 0, 2, 0],
# ])
# child.eval_paths(actions) # 0.94

# states = Variable(torch.randn(controller_paths_per_step * 5, state_dim))
# actions, log_probs, entropies = controller(states)
# rewards = child.eval_paths(actions, n=1)

# a = to_numpy(actions)
# sel = (a[:,2] == 2) & (a[:,3] == 1) & (a[:,4] == 1)
# a[sel]

# <<

class Logger(object):
    def __init__(self, outpath):
        self.log_file    = open(outpath + '.log', 'w')
        self.action_file = open(outpath + '.actions', 'w')
        
        self.history = []
    
    def log(self, step, child, rewards, actions, mode):
        rewards, actions = to_numpy(rewards), to_numpy(actions)
        record = OrderedDict([
            ("mode",             mode),
            ("step",             step),
            ("mean_reward",      round(float(rewards.mean()), 5)),
            ("max_reward",       round(float(rewards.max()), 5)),
            ("mean_actions",     list(actions.mean(axis=0))),
            ("records_seen",     dict(child.records_seen)),
        ])
        print(json.dumps(record), file=self.log_file)
        self.history.append(record)
        
        for reward, action in zip(rewards.squeeze(), actions):
            line = [mode, step, reward] + list(action)
            print('\t'.join(map(str, line)), file=self.action_file)
        
        self.log_file.flush()
        self.action_file.flush()
        
    def close(self):
        self.log_file.close()
        self.action_file.close()

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
    
    states = Variable(torch.randn(args.controller_eval_paths_per_epoch, state_dim))
    actions, log_probs, entropies = controller(states)
    rewards = child.eval_paths(actions, mode='test')
    
    logger.log(total_controller_steps, child, rewards, actions, mode='test')

logger.close()