#!/usr/bin/env python

"""
    ftop_controller.py
"""

from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .base_controller import Controller
from .controller_helpers import sample_softmax, sample_bernoulli, prep_logits

class FTopStep(nn.Module):
    step_length = 1
    def __init__(self, num_ops, hidden_dim):
        super(FTopStep, self).__init__()
        
        self.decoder = nn.Linear(hidden_dim, num_ops)
        self.emb     = nn.Embedding(num_ops, hidden_dim)
        
    def init_weights(self):
        self.decoder.bias.data.fill_(0)


class FTopLSTMController(Controller, nn.Module):
    def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, n_input_nodes=1, 
        temperature=1, clip_logits=-1, opt_params={}, cuda=False):
        
        super(FTopLSTMController, self).__init__()
        
        self.input_dim       = input_dim
        self.output_length   = output_length
        self.output_channels = output_channels
        self.hidden_dim      = hidden_dim
        self.temperature     = temperature
        self.clip_logits     = clip_logits
        
        self.state_encoder   = nn.Linear(input_dim, hidden_dim) # maps observations to lstm dim
        self.lstm_cell       = nn.LSTMCell(hidden_dim, hidden_dim)
        
        steps = []
        for step_idx in range(output_length):
            step = FTopStep(num_ops=output_channels, hidden_dim=hidden_dim)
            step.init_weights()
            steps.append(step)
        
        self.init_weights()
        
        self.steps = nn.ModuleList(steps)
        self.step_length = FTopStep.step_length
        
        self._cuda    = cuda
        self.baseline = None
        self.opt      = torch.optim.Adam(self.parameters(), **opt_params)
        
        if cuda:
            self.cuda()
    
    def init_weights(self, init_bound=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_bound, init_bound)
    
    def __call__(self, states, fixed_actions=None):
        lstm_state = (
            Variable(torch.zeros(states.shape[0], self.hidden_dim)),
            Variable(torch.zeros(states.shape[0], self.hidden_dim)),
        )
        
        if self._cuda:
            states = states.cuda()
            lstm_state = (lstm_state[0].cuda(), lstm_state[1].cuda())
        
        lstm_inputs = self.state_encoder(states)
        
        all_actions, all_action_log_probs, all_entropies = [], [], []
        for step_idx in range(self.output_length):
            step = self.steps[step_idx]
            
            decoder, emb = step.decoder, step.emb
            
            offset = step_idx * self.step_length
            
            lstm_state = self.lstm_cell(lstm_inputs, lstm_state)
            logits = decoder(lstm_state[0])
            
            logits = prep_logits(logits, temperature=self.temperature, clip_logits=self.clip_logits)
            
            actions, action_log_probs, entropy = sample_softmax(
                logits=logits,
                fixed_action=fixed_actions[:,offset] if fixed_actions is not None else None
            )
            
            all_actions.append(actions)
            all_action_log_probs.append(action_log_probs)
            all_entropies.append(entropy)
            
            lstm_inputs = emb(actions)
            
        all_actions          = torch.stack(all_actions, dim=-1)
        all_action_log_probs = torch.cat(all_action_log_probs, dim=-1)
        all_entropies        = torch.cat(all_entropies, dim=-1)
        
        return all_actions, all_action_log_probs, all_entropies