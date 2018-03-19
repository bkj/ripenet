#!/usr/bin/env python

"""
    controller_helpers.py
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def sample_softmax(logits, fixed_action=None):
    # Sample action
    probs = F.softmax(logits, dim=-1)
    if fixed_action is None:
        actions = probs.multinomial(num_samples=1).squeeze()
    else:
        actions = fixed_action.contiguous().squeeze()
    
    # Compute log probability
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.view(-1, 1))
    
    # Compute entropy
    entropy = -(probs * log_probs).sum(dim=1)
    
    return actions.long(), action_log_probs.float(), entropy.float()


def sample_bernoulli(logits, fixed_action=None):
    # Sample action
    probs = F.sigmoid(logits)
    
    # Encode/decode actions as integers (here using prime factorization)
    prime_mask  = _PRIMES[:probs.shape[1]].expand_as(probs)
    if fixed_action is None:
        actions = probs.bernoulli().float()
        int_actions = (prime_mask * actions) + (actions == 0).float() # 
        int_actions = int_actions.prod(dim=-1).long()
    else:
        int_actions = fixed_action.contiguous().squeeze()
        actions     = (int_actions.float().view(-1, 1).remainder(prime_mask) == 0).float()
    
    # Compute log probability
    log_probs = F.logsigmoid(logits)
    action_log_probs = (log_probs * actions).sum(dim=-1)
    
    # Compute entropy
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return actions.long(), int_actions.long(), action_log_probs.float(), entropy.float()

def prep_logits(logits, temperature=1.0, clip_logits=-1):
    if clip_logits == -1:
        return logits / temperature
    elif clip_logits > 0:
        return clip_logits * F.tanh(logits / temperature)
    else:
        raise Exception('prep_logits: illegal clip_logits value of %f' % clip_logits)
