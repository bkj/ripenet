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
f = '_results/cub/cub_3/test.actions'

df = pd.read_csv(f, header=None, sep='\t')
df.columns = ['mode', 'epoch', 'score'] + list(range(df.shape[1] - 4)) + ['aid']

dfs = df[(df.epoch > 50)]
for idx, i in enumerate(np.unique(dfs.aid)):
    print(i)
    tmp = dfs[dfs.aid == i]
    _ = plt.scatter(tmp.epoch, tmp.score, s=5, alpha=0.25)

_ = plt.grid(alpha=0.25)
_ = plt.ylim(0.3, 0.8)
show_plot()



last_good = df.epoch.max() - (df.epoch.max() % 10) - 2
print("last_good=%f" % last_good)
dfs = df[(df.epoch >= (last_good - 5)) & (df.epoch <= last_good)]
print(dfs.groupby('aid').score.agg({
    'mean' : np.mean,
    'n' : len,
}).sort_values('mean'))
