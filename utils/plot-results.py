#!/usr/bin/env python

import sys
import json
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

array = np.array

all_data = []
paths = sorted(sys.argv[1:])
for p in paths:
    data = list(map(json.loads, open(p).readlines()))
    
    # val
    sub = [d for d in data if d['mode'] == 'val']
    mean_reward = [d['mean_reward'] for d in sub]
    try:
        controller_step = [d['controller_step'] for d in sub]
    except:
        controller_step = [d['step'] for d in sub]
    
    _ = plt.plot(controller_step, mean_reward, label=p, alpha=0.25 + 0.75 * (p == paths[-1]))
    
    # test
    sub = [d for d in data if d['mode'] == 'test']
    mean_reward = [d['mean_reward'] for d in sub]
    try:
        controller_step = [d['controller_step'] for d in sub]
    except:
        controller_step = [d['step'] for d in sub]
    
    _ = plt.plot(controller_step, mean_reward, label=p, alpha=0.25 + 0.75 * (p == paths[-1]))

# _ = plt.legend(loc='lower right')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.ylim(0.95, 1.0)
_ = plt.xlim(0, 500)

for i in np.linspace(0.95, 1.0, 6):
    _ = plt.axhline(i, c='grey', alpha=0.1)

show_plot()