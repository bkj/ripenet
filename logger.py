#!/usr/bin/env python

"""
    logger.py
"""

from __future__ import division, print_function

import os
import json
import pandas as pd
from collections import OrderedDict
from basenet.helpers import to_numpy

class Logger():
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
    
    def log(self, epoch, rewards, actions, step_results=None, mode='train', extra=None):
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
        
        if extra is not None:
            for k,v in extra.items():
                assert k not in record, "Logger: k in record.keys()"
                record[k] = v
        
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