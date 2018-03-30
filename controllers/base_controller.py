#!/usr/bin/env python

"""
    base_controller.py
"""

from __future__ import print_function, absolute_import

import copy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .controller_helpers import sample_softmax, sample_bernoulli, prep_logits

# --
# Controllers

class Controller(nn.Module):
    def __init__(self, *args, decay=0.9, z_weight=0.0, **kwargs):
        self.decay = decay
        self.z_weight = z_weight
        
    def get_advantages(self, rewards):
        if self.baseline is None:
            self.baseline = rewards.mean()
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * rewards
        
        ema_advantage = rewards - self.baseline                    # advantage over EMA
        z_advantage   = (rewards - rewards.mean()) / rewards.std() # advantage over rest of batch
        
        return ((1 - self.z_weight) * ema_advantage) + (self.z_weight * z_advantage)
    
    def reinforce_step(self, rewards, log_probs, entropies, entropy_penalty=0.0):
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
            entropy_loss = entropy_penalty * entropies.sum()
            loss -= entropy_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 40) # !! Is this a reasonable value?
        self.opt.step()
        
        return {
            "advantages_mean" : advantages.mean(),
            "loss"            : loss,
            "entropy_loss"    : entropy_loss if entropies is not None else None,
        }
    
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
            # torch.nn.utils.clip_grad_norm(self.parameters(), 40) # !! Is this a reasonable value?
            self.opt.step()

# --
# Simple MLP controller

class MLPController(Controller):
    def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, temperature=1, 
        opt_params={}, cuda=False, **kwargs):
        
        super().__init__(**kwargs)
        
        self.decay = decay
        self.z_weight = z_weight
        
        self.output_length   = output_length
        self.output_channels = output_channels
        self.temperature     = temperature
        
        self.model = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_length * output_channels)
        ])
        
        self._cuda = cuda
        self.baseline = None
        self.opt = torch.optim.Adam(self.parameters(), **opt_params)
        
        if cuda:
            self.cuda()
            
    def __call__(self, states, fixed_actions=None):
        if self._cuda:
            self.states = states.cuda()
        
        logits = self.model(states)
        logits = logits.view(-1, self.output_channels)
        
        logits = prep_logits(logits, self.temperature)
        
        actions, action_log_probs, entropy = sample_softmax(
            logits=logits,
            fixed_action=fixed_actions if fixed_actions is not None else None
        )
        
        actions          = actions.view(-1, self.output_length)
        action_log_probs = action_log_probs.view(-1, self.output_length)
        entropy          = entropy.view(-1, self.output_length)
        
        return actions, action_log_probs, entropy

# --
# Simple LSTM controller

class BasicStep(nn.Module):
    def __init__(self, output_channels, hidden_dim):
        super().__init__()
        
        self.decoder = nn.Linear(hidden_dim, output_channels)
        self.emb = nn.Embedding(output_channels, hidden_dim)
    
    def init_weights(self):
        self.decoder.bias.data.fill_(0)


class LSTMController(Controller):
    def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, temperature=1, 
        opt_params={}, cuda=False, **kwargs):
        
        super().__init__(**kwargs)
        
        self.decay = decay
        self.z_weight = z_weight
        
        self.input_dim       = input_dim
        self.output_length   = output_length
        self.output_channels = output_channels
        self.hidden_dim      = hidden_dim
        self.temperature     = temperature
        
        self.state_encoder = nn.Linear(input_dim, hidden_dim) # maps observations to lstm dim
        self.lstm_cell     = nn.LSTMCell(hidden_dim, hidden_dim)
        
        steps = []
        for _ in range(output_length):
            step = BasicStep(output_channels=output_channels, hidden_dim=hidden_dim)
            step.init_weights()
            steps.append(step)
        
        self.steps = nn.ModuleList(steps)
        
        self._cuda    = cuda
        self.baseline = None
        self.opt = torch.optim.Adam(self.parameters(), **opt_params)
        
        if cuda:
            self.cuda()
    
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
            
            lstm_state = self.lstm_cell(lstm_inputs, lstm_state)
            logits = step.decoder(lstm_state[0])
            
            logits = prep_logits(logits, self.temperature)
            
            actions, log_probs, entropy = sample_softmax(
                logits=logits,
                fixed_action=fixed_actions[:,step_idx] if fixed_actions is not None else None
            )
            
            all_actions.append(actions)
            all_action_log_probs.append(log_probs)
            all_entropies.append(entropy)
            
            lstm_inputs = step.emb(actions)
        
        all_actions          = torch.stack(all_actions, dim=1)
        all_action_log_probs = torch.cat(all_action_log_probs, dim=-1)
        all_entropies        = torch.cat(all_entropies, dim=-1)
        
        return all_actions, all_action_log_probs, all_entropies