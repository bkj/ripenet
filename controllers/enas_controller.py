#!/usr/bin/env python

"""
    enas_controller.py
"""

from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .base_controller import Controller
from .controller_helpers import sample_softmax, sample_bernoulli, prep_logits

class MicroStep(nn.Module):
    step_length = 4
    def __init__(self, num_ins, num_ops, hidden_dim):
        super().__init__()
        
        self.decoder_in_left = nn.Linear(hidden_dim, num_ins)
        self.decoder_op_left = nn.Linear(hidden_dim, num_ops)
        self.emb_in_left     = nn.Embedding(num_ins, hidden_dim)
        self.emb_op_left     = nn.Embedding(num_ops, hidden_dim)
        
        self.decoder_in_right = nn.Linear(hidden_dim, num_ins)
        self.decoder_op_right = nn.Linear(hidden_dim, num_ops)
        self.emb_in_right     = nn.Embedding(num_ins, hidden_dim)
        self.emb_op_right     = nn.Embedding(num_ops, hidden_dim)
        
    def init_weights(self):
        self.decoder_in_left.bias.data.fill_(0)
        self.decoder_op_left.bias.data.fill_(0)
        self.decoder_in_right.bias.data.fill_(0)
        self.decoder_op_right.bias.data.fill_(0)


class MicroLSTMController(Controller):
    def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, n_input_nodes=1, 
        temperature=1, clip_logits=-1, opt_params={}, cuda=False, **kwargs):
        """
            input_dim:       dimension of states
            output_length:   number of cells
            output_channels: number of operations
            hidden_dim:      dimension of internal representations
            n_input_nodes:   number of input nodes
            temperature:     temperature to scale logits (higher -> more entropy)
            clip_logits:     if > 0, clip logits w/ tanh and scale to this size
            opt_params:      optimizer parameters
        """
        
        super().__init__(**kwargs)
        
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
            step = MicroStep(num_ins=step_idx + n_input_nodes, num_ops=output_channels, hidden_dim=hidden_dim)
            step.init_weights()
            steps.append(step)
        
        self.init_weights()
        
        self.steps = nn.ModuleList(steps)
        self.step_length = MicroStep.step_length
        
        self._cuda    = cuda
        self.baseline = None
        self.opt = torch.optim.Adam(self.parameters(), **opt_params)
        
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
            
            layers = [
                (step.decoder_in_left,  step.emb_in_left),  # left input
                (step.decoder_in_right, step.emb_in_right), # right input
                (step.decoder_op_left,  step.emb_op_left),  # left op
                (step.decoder_op_right, step.emb_op_right), # right op
            ]
            offset = step_idx * self.step_length
            for (decoder, emb) in layers:
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
                offset += 1
            
        all_actions          = torch.stack(all_actions, dim=-1)
        all_action_log_probs = torch.cat(all_action_log_probs, dim=-1)
        all_entropies        = torch.cat(all_entropies, dim=-1)
        
        return all_actions, all_action_log_probs, all_entropies


# --
# ENAS Macro CNN controller
# !! This should work, but haven't used it yet 

# _PRIMES = Variable(torch.FloatTensor([2, 3, 5, 7, 11, 13, 17, 19]))
# assert _PRIMES.prod().data[0] < 2 ** 32

# class MacroStep(nn.Module):
#     step_length = 2
#     def __init__(self, num_ins, num_ops, hidden_dim):
#         super(MacroStep, self).__init__()
        
#         self.decoder_in = nn.Linear(hidden_dim, num_ins)
#         self.decoder_op = nn.Linear(hidden_dim, num_ops)
        
#         self.emb_in = nn.Linear(num_ins, hidden_dim, bias=False)
#         self.emb_op = nn.Embedding(num_ops, hidden_dim)
    
#     def init_weights(self):
#         self.decoder_in.bias.data.fill_(0)
#         self.decoder_op.bias.data.fill_(0)


# class MacroLSTMController(Controller, nn.Module):
#     def __init__(self, input_dim=32, output_length=4, output_channels=2, hidden_dim=32, temperature=1, opt_params={}, cuda=False):
#         super(MacroLSTMController, self).__init__()
        
#         assert len(_PRIMES) >= output_channels, "len(_PRIMES) > output_channels -- switch to binary coding!"
        
#         self.input_dim       = input_dim
#         self.output_length   = output_length
#         self.output_channels = output_channels
#         self.hidden_dim      = hidden_dim
#         self.temperature     = temperature
        
#         self.state_encoder   = nn.Linear(input_dim, hidden_dim) # maps observations to lstm dim
#         self.lstm_cell       = nn.LSTMCell(hidden_dim, hidden_dim)
        
#         steps = []
#         for step_idx in range(output_length):
#             step = MacroStep(num_ins=step_idx + 1, num_ops=output_channels, hidden_dim=hidden_dim)
#             step.init_weights()
#             steps.append(step)
        
#         self.steps = nn.ModuleList(steps)
#         self.step_length = MacroStep.step_length
        
#         self._cuda    = cuda
#         self.baseline = None
#         self.opt = torch.optim.Adam(self.parameters(), **opt_params)
        
#         if cuda:
#             self.cuda()
    
#     def __call__(self, states, fixed_actions=None):
#         lstm_state = (
#             Variable(torch.zeros(states.shape[0], self.hidden_dim)),
#             Variable(torch.zeros(states.shape[0], self.hidden_dim)),
#         )
        
#         if self._cuda:
#             states = states.cuda()
#             lstm_state = (lstm_state[0].cuda(), lstm_state[1].cuda())
        
#         lstm_inputs = self.state_encoder(states)
        
#         all_actions, all_action_log_probs, all_entropies = [], [], []
#         for step_idx in range(self.output_length):
#             step = self.steps[step_idx]
#             offset = step_idx * self.step_length
#             # --
#             # Sample inputs
            
#             lstm_state = self.lstm_cell(lstm_inputs, lstm_state)
#             logits = step.decoder_in(lstm_state[0])
            
#             actions, int_actions, action_log_probs, entropy = sample_bernoulli(
#                 logits=logits,
#                 temperature=self.temperature, 
#                 fixed_action=fixed_actions[:,offset] if fixed_actions is not None else None
#             )
            
#             all_actions.append(int_actions) # Record numeric encoding of k-hot vector
#             all_action_log_probs.append(action_log_probs)
#             all_entropies.append(entropy)
            
#             lstm_inputs = step.emb_in(actions.float()) # Embedding of k-hot vector
            
#             # --
#             # Sample ops
            
#             lstm_state = self.lstm_cell(lstm_inputs, lstm_state)
#             logits     = step.decoder_op(lstm_state[0])
            
#             actions, action_log_probs, entropy = sample_softmax(
#                 logits=logits,
#                 temperature=self.temperature, 
#                 fixed_action=fixed_actions[:,2 * step_idx + 1] if fixed_actions is not None else None
#             )
            
#             all_actions.append(actions)
#             all_action_log_probs.append(action_log_probs)
#             all_entropies.append(entropy)
            
#             lstm_inputs = step.emb_op(actions)
            
#         all_actions          = torch.stack(all_actions, dim=-1)
#         all_action_log_probs = torch.cat(all_action_log_probs, dim=-1)
#         all_entropies        = torch.cat(all_entropies, dim=-1)
        
#         return all_actions, all_action_log_probs, all_entropies