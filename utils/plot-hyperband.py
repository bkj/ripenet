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
    for k in data[0]['mean_reward'].keys():
        mean_rewards = [d['mean_reward'][k] for d in data]
        controller_step = [d['records_seen']['train'] for d in data]
        _ = plt.plot(controller_step, mean_rewards, label=p, alpha=0.25)


_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.grid(c='grey', alpha=0.25)
show_plot()