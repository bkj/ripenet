#!/usr/bin/env python

"""
    logger.py
"""

from __future__ import division, print_function

import os
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


class HyperbandLogger():
    def __init__(self, outpath, action_buffer_length=5000):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        self.log_files = {
            "train"      : open(os.path.join(outpath, 'train.log'), 'w'),
            "val"        : open(os.path.join(outpath, 'val.log'), 'w'),
            "test"       : open(os.path.join(outpath, 'test.log'), 'w'),
            
            "controller" : open(os.path.join(outpath, 'controller.log'), 'w'),
        }
        
        self.action_files = {
            "train" : open(os.path.join(outpath, 'train.actions'), 'w'),
            "val"   : open(os.path.join(outpath, 'val.actions'), 'w'),
            "test"  : open(os.path.join(outpath, 'test.actions'), 'w'),
        }
    
    def close(self):
        for v in self.log_files.values():
            v.close()
        
        for v in self.action_files.values():
            v.close()
    
    def log(self, epoch, rewards, actions, mode='train'):
        rewards       = to_numpy(rewards).squeeze()
        actions       = to_numpy(actions).squeeze()
        action_hashes = [str(a) for a in actions]
        
        # --
        # Write log
        
        mean_rewards  = pd.Series(rewards).groupby(action_hashes).mean().to_dict()
        max_rewards   = pd.Series(rewards).groupby(action_hashes).max().to_dict()
        
        log_file = self.log_files[mode]
        
        record = OrderedDict([
            ("mode",              mode),
            ("step",              epoch),
            ("mean_reward",       mean_rewards),
            ("max_reward",        max_rewards),
            ("mean_train_reward", mean_rewards),
            ("max_train_reward",  max_rewards),
        ])
        print(json.dumps(record), file=log_file)
        log_file.flush()
        
        # --
        # Write actions
        
        action_file = self.action_files[mode]
        for reward, action, action_hash in zip(rewards, actions, action_hashes):
            line = [mode, epoch, round(float(reward), 5)] + list(action) + [action_hash]
            line = '\t'.join(map(str, line))
            print(line, file=action_file)
        
        action_file.flush()
    
    def controller_log(self, epoch, controller_update):
        log_file = self.log_files['controller']
        
        for record in controller_update:
            record['step'] = epoch
            print(json.dumps(record), file=log_file)
        
        log_file.flush()