import pandas as pd
import numpy as np
from rsub import *
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'rainbow'


df = pd.read_csv('./_results/hyperband/hyperband.11.train_actions', header=None, sep='\t')

df['aid'] = list(map(str, np.array(df[list(range(3, 11))])))

cnts = pd.value_counts(df.aid)
cnts = cnts[np.argsort(cnts.index)]

c = np.floor(255 / cnts.shape[0])

window = 256
for idx, i in enumerate(cnts.index):
    tmp = df[2][df.aid == i]
    tmp = tmp.rolling(window).mean()
    _ = plt.plot(tmp, alpha=0.5, label=i, c=plt.cm.rainbow(int(c * idx)))

# plt.subplots_adjust(left=0.4)
# _ = plt.legend(loc='center left', bbox_to_anchor=(-0.7, 0.5), cmap=)

show_plot()
