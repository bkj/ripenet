import pandas as pd
import numpy as np
from rsub import *
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'rainbow'


dfa = pd.read_csv('./_results/hyperband/hyperband.12.actions', header=None, sep='\t')
dfb = pd.read_csv('./_results/hyperband/hyperband.13.actions', header=None, sep='\t')

dfa = dfa.head(dfb.shape[0])

dfa['aid'] = list(map(str, np.array(dfa[list(range(3, 11))])))
dfb['aid'] = list(map(str, np.array(dfb[list(range(3, 11))])))

cnts = pd.value_counts(dfa.aid)
cnts = cnts[np.argsort(cnts.index)]

c = np.floor(255 / cnts.shape[0])

for idx, i in enumerate(cnts.index):
    tmp = dfa[2][dfa.aid == i]
    _ = plt.plot(tmp, alpha=0.5, label=i, c='gray')
    
    tmp = dfb[2][dfb.aid == i]
    _ = plt.plot(tmp, alpha=0.5, label=i, c=plt.cm.rainbow(2 * int(c * idx)))

_ = plt.ylim(0, 1)
show_plot()

