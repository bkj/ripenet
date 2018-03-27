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
f = '_results/hyperband/hyperband.resample_4/train.actions'

df = pd.read_csv(f, header=None, sep='\t')
df.columns = ['mode', 'epoch', 'score'] + range(df.shape[1] - 4) + ['aid']

for idx, i in enumerate(np.unique(df.aid)):
    tmp = df[df.aid == i]
    tmp.score = tmp.score.rolling(4).mean()
    _ = plt.plot(tmp.epoch, tmp.score, alpha=0.5, label=i)

# _ = plt.ylim(0.85, 0.95)
_ = plt.xlim(100, 1000)
_ = plt.grid(alpha=0.25)
_ = plt.yticks(list(plt.yticks()[0]) + list(np.arange(0.9, 1.0, 0.01)))
show_plot()

# --

N = 500
g50 = df.groupby('epoch').score.quantile(0.50)
m50 = df.groupby('epoch').score.quantile(0.50).cummax()
m75 = df.groupby('epoch').score.quantile(0.75)
m95 = df.groupby('epoch').score.quantile(0.95).cummax()
m100 = df.groupby('epoch').score.quantile(1.0).cummax()
_ = plt.plot(g50.tail(N))
_ = plt.plot(m50.tail(N))
_ = plt.plot(m75.tail(N))
_ = plt.plot(m95.tail(N))
_ = plt.plot(m100.tail(N))
_ = plt.ylim(0.85, 0.95)
# _ = plt.xlim(200, 500)
_ = plt.grid(alpha=0.25)
_ = plt.yticks(list(plt.yticks()[0]) + list(np.arange(0.9, 1.0, 0.01)))
show_plot()