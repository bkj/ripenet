#!/usr/bin/env python

import os
import sys
import json
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

all_data = []
paths = sorted(sys.argv[1:])
for p in paths:
    data = list(map(json.loads, open(p).readlines()))
    
    mean_reward = [np.mean(d['mean_reward'].values()) for d in data]
    controller_step = [d['step'] for d in data]
    
    _ = plt.plot(controller_step, mean_reward, label=os.path.basename(p), alpha=0.25)


_ = plt.legend(loc='lower right')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')

for i in np.linspace(0.90, 1.0, 6):
    _ = plt.axhline(i, c='grey', alpha=0.1)

show_plot()
