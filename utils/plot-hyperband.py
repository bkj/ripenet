#!/usr/bin/env python

import sys
import json
import numpy as np
import pandas as pd
from rsub import *
from matplotlib import pyplot as plt

array = np.array

all_data = []
paths = sorted(sys.argv[1:])

for p in paths:
    data = list(map(json.loads, open(p).readlines()))
    for k in data[0]['mean_reward'].keys():
        mean_rewards = [d['mean_reward'].get(k, np.nan) for d in data if d['mode'] == 'test']
        mean_rewards = np.array(pd.Series(mean_rewards).rolling(8).mean())
        controller_step = [d['records_seen']['train'] for d in data if d['mode'] == 'test']
        _ = plt.plot(controller_step, mean_rewards, label=p, alpha=0.5)

_ = plt.ylim(0, 1)
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')
_ = plt.yticks(np.arange(0, 1, 0.1))
_ = plt.grid(c='grey', alpha=0.25)
show_plot()