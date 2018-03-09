#!/usr/bin/env python

"""
    logger.py
"""

from __future__ import division, print_function

import json
import pandas as pd
from collections import OrderedDict, deque
from basenet.helpers import to_numpy

class Logger(object):
    def __init__(self, outpath, action_buffer_length=5000):
        self.log_file    = open(outpath + '.log', 'w')
        self.action_file = open(outpath + '.actions', 'w')
        self.train_action_file = open(outpath + '.train_actions', 'w')
        
        self.history = []
        self.action_buffer_length = action_buffer_length
        self.actions = deque(maxlen=action_buffer_length)
    
    def log(self, step, child, rewards, actions, train_rewards, train_actions, mode):
        rewards, actions = to_numpy(rewards), to_numpy(actions)
        record = OrderedDict([
            ("mode",                   mode),
            ("step",                   step),
            ("mean_reward",            round(float(rewards.mean()), 5)),
            ("max_reward",             round(float(rewards.max()), 5)),
            ("mean_train_reward",      round(float(train_rewards.mean()), 5)) if train_rewards is not None else None,
            ("max_train_reward",       round(float(train_rewards.max()), 5)) if train_rewards is not None else None,
            ("controller_convergence", self.controller_convergence),
            ("records_seen",           dict(child.records_seen)),
        ])
        print(json.dumps(record), file=self.log_file)
        
        self.history.append(record)
        
        for reward, action in zip(rewards.squeeze(), actions):
            line = [mode, step, round(float(reward), 5)] + list(action)
            print('\t'.join(map(str, line)), file=self.action_file)
            self.actions.append(str(action))
        
        self.log_file.flush()
        self.action_file.flush()
        
    def close(self):
        self.log_file.close()
        self.action_file.close()
    
    @property
    def controller_convergence(self):
        actions = list(self.actions)
        if len(actions) == self.action_buffer_length:
            return pd.value_counts(actions).iloc[0] / len(actions)
        else:
            return 0.0


class HyperbandLogger(Logger):
    def log(self, step, child, rewards, actions, train_rewards, train_actions, mode):
        rewards = to_numpy(rewards).squeeze()
        actions = to_numpy(actions)
        action_hashes = [str(a) for a in actions]
        mean_rewards  = pd.Series(rewards).groupby(action_hashes).mean().to_dict()
        max_rewards   = pd.Series(rewards).groupby(action_hashes).max().to_dict()
        
        if train_rewards is not None:
            train_rewards = to_numpy(train_rewards).squeeze()
            train_actions = to_numpy(train_actions).squeeze()
            train_action_hashes = [str(a) for a in train_actions]
            train_mean_rewards  = pd.Series(train_rewards).groupby(train_action_hashes).mean().to_dict()
            train_max_rewards   = pd.Series(train_rewards).groupby(train_action_hashes).max().to_dict()
        else:
            train_mean_rewards, train_max_rewards = None, None
        
        record = OrderedDict([
            ("mode",              mode),
            ("step",              step),
            ("mean_reward",       mean_rewards),
            ("max_reward",        max_rewards),
            ("mean_train_reward", train_mean_rewards),
            ("max_train_reward",  train_max_rewards),
            ("records_seen",      dict(child.records_seen)),
        ])
        print(json.dumps(record), file=self.log_file)
        
        self.history.append(record)
        
        for reward, action in zip(rewards, actions):
            line = [mode, step, round(float(reward), 5)] + list(action)
            print('\t'.join(map(str, line)), file=self.action_file)
        
        if train_rewards is not None:
            for reward, action in zip(train_rewards, train_actions):
                line = [mode, step, round(float(reward), 5)] + list(action)
                print('\t'.join(map(str, line)), file=self.train_action_file)
        
        self.log_file.flush()
        self.action_file.flush()
