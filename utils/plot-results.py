#!/usr/bin/env python

import sys
import json
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

array = np.array

all_data = []
for p in sys.argv[1:]:
    try:
        data = map(eval, open(p).readlines())
        data = [d['mean_reward'] for d in data]
        _ = plt.plot(data, alpha=0.75, label=p)
    except:
        pass

_ = plt.legend(loc='lower right')
_ = plt.axhline(0.9, c='grey', alpha=0.1)
_ = plt.axhline(0.95, c='grey', alpha=0.1)
show_plot()