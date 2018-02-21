#!/usr/bin/env python

import sys
import json
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

array = np.array

all_data = []
for p in sys.argv[1:]:
    # try:
        data = list(map(json.loads, open(p).readlines()))
        mean_reward = [d['mean_reward'] for d in data]
        try:
            controller_step = [d['controller_step'] for d in data]
        except:
            controller_step = [d['step'] for d in data]
        _ = plt.plot(controller_step, mean_reward, label=p, alpha=0.75)
    # except:
        # pass

# _ = plt.legend(loc='lower right')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.axhline(0.9, c='grey', alpha=0.1)
_ = plt.axhline(0.95, c='grey', alpha=0.1)
show_plot()