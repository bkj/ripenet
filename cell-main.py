#!/usr/bin/env python

"""
    main.py
"""

import os
import sys
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from controllers import MicroLSTMController
from children import LazyChild
from workers import CellWorker
from data import make_cifar_dataloaders

from basenet.helpers import to_numpy

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=str, default='lazy_child', choices=['lazy_child'])
    parser.add_argument('--pretrained-path', type=str, default='./pretrained_models/cell_worker-50.weights')
    parser.add_argument('--train-size', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

args = parse_args()

# --
# Parameterss

state_dim = 32

child_steps_per_iter = 100

controller_steps_per_iter = 50
controller_paths_per_step = 100

controller_candidates_per_eval = 100

output_length   = 2 # len(CellBlock.nodes)
output_channels = 8 # len(CellBlock.op_fns)

n_iters = 10

# --
# Initialize controller

controller_kwargs = {
    "input_dim" : state_dim,
    "output_length" : output_length,
    "output_channels" : output_channels,
}


controller = MicroLSTMController(**controller_kwargs)

# --
# Initialize child

dataloaders = make_cifar_dataloaders(train_size=args.train_size,  download=False, seed=args.seed, num_workers=0)

worker = CellWorker().cuda()
if args.pretrained_path is not None:
    print('main.py: loading pretrained model %s' % args.pretrained_path, file=sys.stderr)
    worker.load_state_dict(torch.load(args.pretrained_path))
else:
    raise Exception

# print('worker ->', worker, file=sys.stderr)

if args.child == 'lazy_child':
    child = LazyChild(worker=worker, dataloaders=dataloaders)
else:
    raise Exception('main.py: unknown child %s' % args.child, file=sys.stderr)

# --
# Run

history = []
for iter in range(n_iters):
    print('-' * 50, file=sys.stderr)
    
    # # --
    # # Train child
    
    # states = Variable(torch.randn(child_steps_per_iter, state_dim))
    # actions, _, _ = controller(states)
    # child.train_paths(actions)
    
    # --
    # Train controller
    
    for controller_step in range(controller_steps_per_iter):
        states = Variable(torch.randn(controller_paths_per_step, state_dim))
        actions, log_probs, entropies = controller(states)
        rewards = child.eval_paths(actions, n=1)
        
        if args.algorithm == 'reinforce':
            controller.reinforce_step(rewards, log_probs=log_probs, entropies=entropies)
        elif args.algorithm == 'ppo':
            controller.ppo_step(rewards, states=states, actions=actions)
        else:
            raise Exception('unknown algorithm %s' % args.algorithm, file=sys.stderr)
        
        history.append({
            "mean_reward"     : to_numpy(rewards).mean(),
            "mean_actions"    : to_numpy(actions).mean(axis=0),
            "controller_step" : len(history),
            "eval_records"    : child.eval_records,
        })
        print(history[-1])
    
    # # --
    # # Eval best architecture
    
    # print('eval=%d' % iter, file=sys.stderr)
    # states = Variable(torch.randn(controller_candidates_per_eval, state_dim))
    # actions, log_probs, entropies = controller(states)
    # rewards = child.eval_paths(actions)
    
    # print("rewards.min    ->", rewards.min(), file=sys.stderr)
    # print("rewards.median ->", rewards.median(), file=sys.stderr)
    # print("rewards.max    ->", rewards.max(), file=sys.stderr)

