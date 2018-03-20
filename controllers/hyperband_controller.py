#!/usr/bin/env python

"""
    hyperband_controller.py
"""

from __future__ import print_function, absolute_import

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .base_controller import Controller
from .controller_helpers import sample_softmax, sample_bernoulli, prep_logits

import re

def parse_arch(arch):
    arch = list(re.sub('[^0-9]','', arch))
    arch = np.array(list(map(int, arch)))
    return arch


# GOOD_ARCH = parse_arch("0002_0112")
# BAD_ARCH  = parse_arch("0012_0112")

# ARCHS = [
#     "0002_0112",
#     "0012_0112",
#     "0022_0112",
#     "0032_0112",
    
#     "0003_0112",
#     "0013_0112",
#     "0023_0112",
#     "0033_0112",

#     "0002_0122",
#     "0012_0122",
#     "0022_0122",
#     "0032_0122",
    
#     "0003_0132",
#     "0013_0132",
#     "0023_0132",
#     "0033_0132",
# ]
# ARCHS = np.vstack([parse_arch(a) for a in ARCHS])

class HyperbandController(object):
    def __init__(self, input_dim=32, output_length=2, output_channels=4, population_size=32, **kwargs):
        self.output_length   = output_length
        self.output_channels = output_channels
        self.population      = self.__sample_population(population_size=population_size)
    
    def __sample_population(self, population_size):
        blocks = []
        for block_id in range(self.output_length):
            in_left  = np.random.choice(block_id + 1, population_size)
            in_right = np.random.choice(block_id + 1, population_size)
            op_left  = np.random.choice(self.output_channels, population_size)
            op_right = np.random.choice(self.output_channels, population_size)
            blocks.append(np.column_stack([in_left, in_right, op_left, op_right]))
        
        blocks = np.hstack(blocks)
        # >>
        # for i in range(population_size - 1):
        #     blocks[i] = GOOD_ARCH
        # <<
        
        return blocks
        
        # >>
        # num_good_blocks = 6
        # num_bad_blocks  = population_size - num_good_blocks
        # return np.vstack(
        #     [GOOD_ARCH for _ in range(num_good_blocks)] + 
        #     [BAD_ARCH for _ in range(num_bad_blocks)]
        # )
        # <<
        
        return ARCHS
    
    def __call__(self, states):
        action_idx = np.random.choice(self.population.shape[0], states.shape[0])
        all_actions = torch.LongTensor(self.population[action_idx])
        return all_actions, None, None


if __name__ == '__main__':
    c = HyperbandController()
    all_actions, _, _ = c(np.arange(10))
    print(all_actions)