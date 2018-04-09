#!/usr/bin/env python

"""
    action-plot.py
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pprint import pprint

from rsub import *
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'rainbow'

# --


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='cub_8')
    return parser.parse_args()

args = parse_args()

# Print config
print('-- %s --' % args.run)
pprint(json.load(open('_results/cub/%s/config' % args.run)))

# Load best actions
log_path = '_results/cub/%s/test.log' % args.run
data = list(map(json.loads, open(log_path).readlines()[:-1]))
data = list(filter(lambda x: 'predict' in x, data))

pred_rewards = np.hstack([sorted(d['mean_reward'].values()) for d in data])
pred_steps = np.hstack([[d['step']] * len(d['mean_reward']) for d in data])


# Plot actions
action_path = '_results/cub/%s/test.actions' % args.run
df = pd.read_csv(action_path, header=None, sep='\t')
df.columns = ['mode', 'epoch', 'score'] + list(range(df.shape[1] - 4)) + ['aid']

dfs = df# [(df.epoch > 50)]
for idx, i in enumerate(np.unique(dfs.aid)):
    tmp = dfs[dfs.aid == i]
    _ = plt.plot(tmp.epoch, tmp.score, alpha=0.25)

q50 = dfs.groupby('epoch').score.apply(lambda x: np.percentile(x, 50))
_ = plt.plot(q50, c='blue')
_ = plt.plot(q50.cummax(), c='red')
_ = plt.grid(alpha=0.25)
# _ = plt.ylim(0.3, 0.8)
# _ = plt.xlim(0, 50)
_ = plt.plot(pred_steps, pred_rewards, c='yellow')
_ = plt.title(args.run)
_ = plt.axhline(0.65, color='grey')
show_plot()



# last_good = (df.epoch.max() + 1) - (df.epoch.max() + 1) % 20 - 1
# print(last_good)
# dfs = df[df.epoch == last_good].tail(32)
# print(dfs.groupby('aid').score.mean().sort_values())

# df.groupby('epoch').aid.apply(lambda x: len(set(x)))

# # >>



# # <<