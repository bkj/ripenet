#!/usr/bin/env python

"""
    test.py
"""

import sys
import numpy as np
from tqdm import tqdm

from rsub import *
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet.helpers import to_numpy

sys.path.append('../')
from controllers import LSTMController, MLPController

# --
# Simple environment

class SimpleEnv(object):
    def __init__(self, output_length):
        n = int(output_length / 2)
        self._payouts = torch.FloatTensor(([1] * n) + ([-1] * n))
    
    def eval(self, actions):
        rewards = self._payouts * actions.cpu().float()
        return rewards.sum(dim=1).view(-1, 1)

# --
# Run

if __name__ == "__main__":
    
    batch_size      = 16
    input_dim       = 32
    entropy_penalty = 0.01
    output_length   = 16
    output_channels = 2
    controller_type = 'lstm'
    step_type       = 'reinforce_step'
    cuda            = True
    
    assert step_type in ['reinforce_step', 'ppo_step']
    
    env = SimpleEnv(output_length=output_length)
    
    if controller_type == 'mlp':
        controller = MLPController(input_dim=input_dim, output_length=output_length, output_channels=output_channels, cuda=cuda)
    elif controller_type == 'lstm':
        controller = LSTMController(input_dim=input_dim, output_length=output_length, output_channels=output_channels, cuda=cuda)
    else:
        raise Exception()
    
    all_rewards, all_actions = [], []
    for _ in tqdm(range(500)):
        states = torch.zeros((batch_size, input_dim)).to(torch.device('cuda'))
        actions, log_probs, entropies = controller(states)
        rewards = env.eval(actions)
        getattr(controller, step_type)(rewards, log_probs, entropies=entropies, entropy_penalty=entropy_penalty)
        
        all_rewards.append(to_numpy(rewards))
        all_actions.append(to_numpy(actions))
        
    all_rewards = np.vstack(all_rewards)
    all_actions = np.vstack(all_actions)
    
    _ = plt.plot(all_rewards.squeeze())
    show_plot()