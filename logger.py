import json
from collections import OrderedDict
from basenet.helpers import to_numpy

class Logger(object):
    def __init__(self, outpath):
        self.log_file    = open(outpath + '.log', 'w')
        self.action_file = open(outpath + '.actions', 'w')
        
        self.history = []
    
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
        
        self.log_file.flush()
        self.action_file.flush()
        
    def close(self):
        self.log_file.close()
        self.action_file.close()