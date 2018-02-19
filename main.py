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

from controllers import MLPController, LSTMController
from children import Child, LazyChild
from workers import MaskWorker
from data import make_cifar_dataloaders

from basenet.helpers import to_numpy

np.set_printoptions(linewidth=120)

# --
# CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'lstm'])
    parser.add_argument('--algorithm', type=str, default='reinforce', choices=['reinforce', 'ppo'])
    parser.add_argument('--child', type=str, default='lazy_child', choices=['lazy_child'])
    return parser.parse_args()

args = parse_args()

# --
# Parameterss

state_dim = 32

child_steps_per_iter = 100

controller_steps_per_iter = 50
controller_paths_per_step = 100

controller_candidates_per_eval = 100

output_length = 12 # Defined by pipenet
output_channels = 2

n_iters = 10

# --
# Initialize controller

controller_kwargs = {
    "input_dim" : state_dim,
    "output_length" : output_length,
    "output_channels" : output_channels,
}

if args.architecture == 'mlp':
    controller = MLPController(**controller_kwargs)
else:
    controller = LSTMController(**controller_kwargs)

# --
# Initialize child

dataloaders = make_cifar_dataloaders(train_size=0.9, download=False, seed=123, num_workers=0)

worker = MaskWorker().cuda()
# worker.init_optimizer(opt=torch.optim.SGD, params=worker.parameters(), lr=0.1, momentum=0.9)
worker.load_state_dict(torch.load('./pretrained_models/sgdr-train0.9/weights'))
print('worker ->', worker, file=sys.stderr)

if args.child == 'lazy_child':
    child = LazyChild(worker=worker, dataloaders=dataloaders)

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

