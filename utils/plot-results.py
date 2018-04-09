#!/usr/bin/env python

from __future__ import division, print_function

import os
import sys
import json
import numpy as np
import pandas as pd
from rsub import *
from matplotlib import pyplot as plt

# --
# IO

run = 'cub_6'
path = '_results/cub/%s/train.log' % run
data = list(map(json.loads, open(path).readlines()[:-1]))

rewards, steps = [], []
predict_rewards, predict_steps = [], []
for d in data:
    for reward in d['mean_reward'].values():
        rewards.append(reward)
        steps.append(d['step'])
    
    if 'predict' in d:
        for reward in d['mean_reward'].values():
            predict_rewards.append(reward)
            predict_steps.append(d['step'])

_ = plt.scatter(steps, rewards, label='noisy', s=5, alpha=0.25)
_ = plt.scatter(predict_steps, predict_rewards, label='topk', s=20, alpha=1, c='red')

for r in sorted(predict_rewards)[-10:]:
    print(r)

print('mean', np.mean(predict_rewards[:-len(predict_rewards) // 2]))

p25 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 25)).rolling(5).mean()
p50 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 50)).rolling(5).mean()
p75 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 75)).rolling(5).mean()
p90 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 90)).rolling(5).mean()

_ = plt.plot(p25, c='red', alpha=0.75)
_ = plt.plot(p50, c='blue', alpha=0.75)
_ = plt.plot(p75, c='green', alpha=0.75)
# _ = plt.plot(p90, c='orange', alpha=0.75)

# _ = plt.xlim(0, 100)
# _ = plt.ylim(0.5, 0.7)

_ = plt.axhline(0.65, c='yellow')
_ = plt.legend(loc='upper left')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.title('reward')
_ = plt.grid(alpha=0.25)
show_plot()



# --
# Controller convergence

action_path = '_results/cub/%s/train.actions' % run
actions = pd.read_csv(action_path, header=None, sep='\t')
actions.columns = ['mode', 'epoch', 'score'] + list(range(actions.shape[1] - 4)) + ['aid']

ucount = actions.groupby('epoch').aid.apply(lambda x: len(set(x)) / len(x))
ucount += np.random.normal(0, 0.01, ucount.shape[0])
_ = plt.plot(ucount, c='red')
_ = plt.ylim(-0.1, 1.1)
_ = plt.axhline(0, c='grey')
_ = plt.axhline(1, c='grey')
_ = plt.title('num_unique_arch / num_arch')
show_plot()
