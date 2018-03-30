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

run = 'reinforce_1'
path = '_results/cub/%s/test.log' % run
action_path = '_results/cub/%s/train.actions' % run

data = list(map(json.loads, open(path).readlines()))

actions = pd.read_csv(action_path, header=None, sep='\t')
actions.columns = ['mode', 'epoch', 'score'] + list(range(actions.shape[1] - 4)) + ['aid']

# --
# Plot

rewards, steps = [], []
for d in data:
    for reward in d['mean_reward'].values():
        rewards.append(reward)
        steps.append(d['step'])

_ = plt.scatter(steps, rewards, label=os.path.basename(path), s=5, alpha=0.25)

p25 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 25))
p50 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 50))
p75 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 75))

_ = plt.plot(p25, c='red', alpha=0.75)
_ = plt.plot(p50, c='blue', alpha=0.75)
_ = plt.plot(p75, c='green', alpha=0.75)

_ = plt.legend(loc='upper left')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.title('reward')

show_plot()


# --
#

ucount = actions.groupby('epoch').aid.apply(lambda x: len(set(x)) / len(x))
_ = plt.plot(ucount, c='red')
_ = plt.ylim(0, 1)
_ = plt.title('num_unique_arch / num_arch')
show_plot()
