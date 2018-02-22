import json
import pandas as pd
from collections import OrderedDict, deque
from basenet.helpers import to_numpy

class Logger(object):
    def __init__(self, outpath, action_buffer_length=2000):
        self.log_file    = open(outpath + '.log', 'w')
        self.action_file = open(outpath + '.actions', 'w')
        
        self.history = []
        self.actions = deque(maxlen=action_buffer_length)
    
    def log(self, step, child, rewards, actions, mode):
        rewards, actions = to_numpy(rewards), to_numpy(actions)
        record = OrderedDict([
            ("mode",             mode),
            ("step",             step),
            ("mean_reward",      round(float(rewards.mean()), 5)),
            ("max_reward",       round(float(rewards.max()), 5)),
            ("mean_actions",     list(actions.mean(axis=0))),
            ("records_seen",     dict(child.records_seen)),
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
    
    def check_actions(self, p=0.75):
        """ could call this to stop if there's not enough variety """
        return pd.value_counts(self.actions).iloc[0] > len(self.actions) * p