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
f = '_results/hyperband/hyperband.resample_0.train_actions'

df = pd.read_csv(f, header=None, sep='\t')
df['aid'] = list(map(str, df[list(range(3, 11))].values))

for idx, i in enumerate(np.unique(df.aid)):
    tmp = df[df.aid == i]
    _ = plt.plot(tmp[1], tmp[2], alpha=0.5, label=i)

_ = plt.ylim(0, 1)
show_plot()

