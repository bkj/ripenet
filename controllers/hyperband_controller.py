#!/usr/bin/env python

"""
    hyperband_controller.py
"""

from __future__ import print_function, absolute_import

import sys
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet.helpers import to_numpy
from .base_controller import Controller
from .controller_helpers import sample_softmax, sample_bernoulli, prep_logits


class HyperbandController(object):
    def __init__(self, input_dim=32, output_length=2, output_channels=4, population_size=32, **kwargs):
        self.output_length   = output_length
        self.output_channels = output_channels
        
        self.population_size = population_size
        self.population      = self.__initialize_population()
        
        template = []
        for block_id in range(output_length):
            template += [
                block_id + 1,         # in_left
                block_id + 1,         # in_right
                self.output_channels, # op_left
                self.output_channels, # op_right
            ]
        
        self.template = np.array(template)
        
    def __initialize_population(self):
        blocks = []
        for block_id in range(self.output_length):
            in_left  = np.random.choice(block_id + 1, self.population_size)
            in_right = np.random.choice(block_id + 1, self.population_size)
            op_left  = np.random.choice(self.output_channels, self.population_size)
            op_right = np.random.choice(self.output_channels, self.population_size)
            blocks.append(np.column_stack([in_left, in_right, op_left, op_right]))
        
        return np.hstack(blocks)
    
    def hyperband_step(self, rewards, resample=False):
        reward_ranking = np.argsort(to_numpy(-rewards).squeeze())
        
        self.population = self.population[reward_ranking]
        self.population = self.population[:int(self.population_size / 2)]
        
        if not resample:
            self.population_size = self.population.shape[0]
        else:
            new_population = []
            for member in self.population:
                new_member = member.copy()
                while (new_member == member).all():
                    idx = np.random.choice(member.shape[0])
                    new_member[idx] = np.random.choice(self.template[idx])
                
                new_population.append(new_member)
            
            self.population = np.column_stack([self.population, new_population]).reshape((self.population.shape[0] * 2, self.population.shape[1]))
        
        print('HyperbandController.hyperband_step: new population (%d) ->\n %s' % (self.population_size, str(self.population)), file=sys.stderr)
    
    def __call__(self, states):
        action_idx = np.random.choice(self.population_size, states.shape[0])
        all_actions = torch.LongTensor(self.population[action_idx])
        return all_actions, None, None


if __name__ == '__main__':
    c = HyperbandController()
    print(to_numpy(c.population))
    print(c.template)
    
    c.hyperband_step(np.arange(c.population_size), resample=True)
    c.population