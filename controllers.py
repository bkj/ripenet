#!/usr/bin/env python

"""
    controllers.py
"""

import sys
import copy
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class Controller(object):
    def get_advantages(self, rewards, decay=0.9):
        if self.baseline is None:
            self.baseline = rewards.mean()
        else:
            self.baseline = decay * self.baseline + (1 - decay) * rewards
        
        return rewards - self.baseline
    
    def reinforce_step(self, rewards, log_probs, entropies=None, entropy_penalty=0.0):
        """ REINFORCE """
        
        if self._cuda:
            rewards = rewards.cuda()
            log_probs = log_probs.cuda()
            if entropies is not None:
                entropies = entropies.cuda()
        
        advantages = self.get_advantages(rewards)
        advantages = Variable(advantages, requires_grad=False)
        advantages = advantages.expand_as(log_probs)
        
        self.opt.zero_grad()
        
        loss = -(log_probs * advantages).sum()
        if entropies is not None:
            loss -= entropy_penalty * entropies.sum()
        
        loss.backward()
        self.opt.step()
    
    def ppo_step(self, rewards, states, actions, entropy_penalty=0.0, clip_eps=0.2, ppo_epochs=4):
        """ Proximal Policy Optimization """
        
        if self._cuda:
            rewards = rewards.cuda()
        
        advantages = self.get_advantages(rewards)
        advantages = Variable(advantages, requires_grad=False)
        
        old = copy.deepcopy(self)
        for ppo_epoch in range(ppo_epochs):
            
            self.opt.zero_grad()
            
            _, log_probs, entropies = self(states, fixed_actions=actions)
            _, old_log_probs, _ = old(states, fixed_actions=actions)
            
            advantages = advantages.expand_as(log_probs)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            loss  = -torch.min(surr1, surr2).mean()
            
            loss -= entropy_penalty * entropies.sum()
            
            loss.backward()
            self.opt.step()

# <<

class MLPController(Controller, nn.Module):
    def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, temperature=1, cuda=False):
        super(MLPController, self).__init__()
        
        self.output_length = output_length
        self.output_channels = output_channels
        self.temperature = temperature
        
        self.model = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_length * output_channels)
        ])
        
        self._cuda = cuda
        self.baseline = None
        self.opt = torch.optim.Adam(self.parameters())
        
        if cuda:
            self.cuda()
            
    def __call__(self, states, fixed_actions=None):
        if self._cuda:
            self.states = states.cuda()
        
        logits = self.model(states)
        logits = logits.view(-1, self.output_channels)
        
        probs   = F.softmax(logits * self.temperature, dim=-1)
        if fixed_actions is None:
            actions = probs.multinomial().squeeze()
        else:
            actions = fixed_actions
        
        log_probs = F.log_softmax(logits * self.temperature, dim=-1)
        action_log_probs = log_probs.gather(1, actions.view(-1, 1))
        
        entropy = -(probs * log_probs).sum(dim=1)
        
        actions = actions.view(-1, self.output_length)
        action_log_probs = action_log_probs.view(-1, self.output_length)
        entropy = entropy.view(-1, self.output_length)
        
        return actions, action_log_probs, entropy


class LSTMController(Controller, nn.Module):
    def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, temperature=1, cuda=False):
        super(LSTMController, self).__init__()
        
        self.input_dim       = input_dim
        self.output_length   = output_length
        self.output_channels = output_channels
        
        self.hidden_dim      = hidden_dim
        self.temperature     = temperature
        
        self.state_encoder   = nn.Linear(input_dim, hidden_dim) # maps observations to lstm dim
        self.state_embedding = nn.Embedding(output_length * output_channels, hidden_dim) # map previous actions to lstm dim
        self.lstm_cell       = nn.LSTMCell(hidden_dim, hidden_dim)
        self.decoders        = nn.ModuleList([nn.Linear(hidden_dim, output_channels)] * output_length)
        
        self.init_weights()
        
        self._cuda    = cuda
        self.baseline = None
        self.opt = torch.optim.Adam(self.parameters())
        
        if cuda:
            self.cuda()
    
    def init_weights(self):
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)
        
        # !! Need to initialize other parameters?
    
    def __call__(self, states, fixed_actions=None):
        lstm_state = (
            Variable(torch.zeros(states.shape[0], self.hidden_dim)),
            Variable(torch.zeros(states.shape[0], self.hidden_dim)),
        )
        
        if self._cuda:
            states = states.cuda()
            lstm_state = (lstm_state[0].cuda(), lstm_state[1].cuda())
        
        lstm_inputs = self.state_encoder(states)
        
        all_actions, all_log_probs, all_entropies = [], [], []
        for step_idx in range(self.output_length):
            
            # Run through LSTM
            lstm_state = self.lstm_cell(lstm_inputs, lstm_state)
            logits = self.decoders[step_idx](lstm_state[0])
            
            # !! Scale logits by temperature and/or tanh as mentioned in paper
            
            # Sample action
            probs   = F.softmax(logits * self.temperature, dim=-1)
            if fixed_actions is None:
                actions = probs.multinomial().squeeze()
            else:
                actions = fixed_actions[:,step_idx].contiguous().squeeze()
            
            all_actions.append(actions)
            
            # Compute log probability
            log_probs = F.log_softmax(logits * self.temperature, dim=-1)
            all_log_probs.append(log_probs.gather(1, actions.view(-1, 1)))
            
            entropy = -(probs * log_probs).sum(dim=1)
            all_entropies.append(entropy)
            
            lstm_inputs = self.state_embedding(actions + self.output_channels * step_idx)
        
        all_actions   = torch.stack(all_actions, dim=1)
        all_log_probs = torch.cat(all_log_probs, dim=-1)
        all_entropies = torch.cat(all_entropies, dim=-1)
        
        return all_actions, all_log_probs, all_entropies