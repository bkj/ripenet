#!/usr/bin/env python

"""
    test.py
"""

import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from controllers import LSTMController, MLPController

from basenet.helpers import to_numpy

from rsub import *
from matplotlib import pyplot as plt

# --
# Simple environment

class SimpleEnv(object):
    def __init__(self, output_length):
        n = int(output_length / 2)
        self._payouts = torch.FloatTensor(([1] * n) + ([-1] * n))
    
    def eval(self, actions):
        rewards = self._payouts * actions.cpu().data.float()
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
    cuda            = True
    
    env = SimpleEnv(output_length=output_length)
    
    if controller_type == 'mlp':
        controller = MLPController(input_dim=input_dim, output_length=output_length, output_channels=output_channels, cuda=cuda)
    elif controller_type == 'lstm':
        controller = LSTMController(input_dim=input_dim, output_length=output_length, output_channels=output_channels, cuda=cuda)
    else:
        raise Exception()
    
    all_rewards, all_actions = [], []
    for _ in tqdm(range(500)):
        states = Variable(torch.zeros((batch_size, input_dim))).cuda()
        actions, log_probs, entropies = controller.sample(states)
        rewards = env.eval(actions)
        controller.step(rewards, log_probs, entropies=entropies, entropy_penalty=entropy_penalty)
        
        all_rewards.append(to_numpy(rewards))
        all_actions.append(to_numpy(actions))
        
    all_rewards = np.vstack(all_rewards)
    all_actions = np.vstack(all_actions)
    
    _ = plt.plot(all_rewards.squeeze())
    show_plot()
