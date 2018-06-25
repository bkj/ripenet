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

run = 'cub_002'
path = '_results/cub/%s/test.log' % run
data = list(map(json.loads, open(path).readlines()[:-1]))

rewards, steps = [], []
predict_rewards, predict_steps = [], []
for d in data[-250:]:
    for reward in d['mean_reward'].values():
        rewards.append(reward)
        steps.append(d['step'])
    
    # if 'predict' in d:
    #     for reward in d['mean_reward'].values():
    #         predict_rewards.append(reward)
    #         predict_steps.append(d['step'])

_ = plt.scatter(steps, rewards, label='noisy', s=5, alpha=0.25)
_ = plt.scatter(predict_steps, predict_rewards, label='topk', s=20, alpha=1, c='red')

for r in sorted(predict_rewards)[-10:]:
    print(r)

p25 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 25))
p50 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 50))
p75 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 75))
p90 = pd.Series(rewards).groupby(steps).apply(lambda x: np.percentile(x, 90))

_ = plt.plot(p25, c='red', alpha=0.75)
_ = plt.plot(p50, c='blue', alpha=0.75)
_ = plt.plot(p75, c='green', alpha=0.75)

_ = plt.axhline(0.65, c='grey')
_ = plt.legend(loc='upper left')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.title('reward')
_ = plt.grid(alpha=0.25)
show_plot()
