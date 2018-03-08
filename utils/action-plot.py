import pandas as pd
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

df = pd.read_csv('./_results/hyperband/hyperband.9.actions', header=None, sep='\t')

df['aid'] = list(map(str, np.array(df[list(range(3, 11))])))

cnts = pd.value_counts(df.aid)
cnts = cnts[np.argsort(cnts.index)]

window = 64
for i in cnts.index:
    tmp = df[2][df.aid == i]
    tmp = tmp.rolling(window).mean()
    _ = plt.plot(tmp, alpha=0.5, label=i)

_ = plt.legend(loc='lower right')
show_plot()
