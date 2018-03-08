#!/usr/bin/env python

import sys
import json
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

array = np.array

all_data = []
paths = sorted(sys.argv[1:])
k_good = '[0 0 0 2 0 1 1 2]'
k_bad = '[0 0 1 1 1 1 3 0]'
for p in paths:
    data = list(map(json.loads, open(p).readlines()))
    mean_reward_good = [d['mean_reward'][k_good] for d in data]
    # mean_reward_bad  = [d['mean_reward'][k_bad] for d in data]
    controller_step = [d['records_seen']['train'] for d in data]
    
    _ = plt.plot(controller_step, mean_reward_good, label=p, alpha=0.25 + 0.75 * (p == paths[-1]))


_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.grid(c='grey', alpha=0.25)
show_plot()