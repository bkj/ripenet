#!/usr/bin/env python

"""
    action-plot.py
"""

import sys
import pandas as pd
import numpy as np

from rsub import *
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'rainbow'

# --

# f = sys.argv[1]
f = '_results/cub/cub_4/test.actions'

df = pd.read_csv(f, header=None, sep='\t')
df.columns = ['mode', 'epoch', 'score'] + list(range(df.shape[1] - 4)) + ['aid']

dfs = df# [(df.epoch > 50)]
for idx, i in enumerate(np.unique(dfs.aid)):
    tmp = dfs[dfs.aid == i]
    _ = plt.plot(tmp.epoch, tmp.score, alpha=0.25)

q75 = dfs.groupby('epoch').score.apply(lambda x: np.percentile(x, 75))
_ = plt.plot(q75, c='blue')
_ = plt.plot(q75.cummax(), c='red')
_ = plt.grid(alpha=0.25)
_ = plt.ylim(0.3, 0.8)
show_plot()



last_good = (df.epoch.max() + 1) - (df.epoch.max() + 1) % 20 - 1
print(last_good)
dfs = df[df.epoch == last_good].tail(32)
print(dfs.groupby('aid').score.mean().sort_values())

df.groupby('epoch').aid.apply(lambda x: len(set(x)))