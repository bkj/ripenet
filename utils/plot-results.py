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
    
    # val
    val_sub = [d for d in data if d['mode'] == 'val']
    mean_reward = [d['mean_reward'] for d in val_sub]
    try:
        controller_step = [d['controller_step'] for d in val_sub]
    except:
        controller_step = [d['step'] for d in val_sub]
    
    print(p)
    _ = plt.plot(controller_step, mean_reward, label=os.path.basename(p) + '_val', alpha=0.25)
    
    # test
    test_sub = [d for d in data if d['mode'] == 'test']
    mean_reward = [d['mean_reward'] for d in test_sub]
    try:
        controller_step = [d['controller_step'] for d in test_sub]
    except:
        controller_step = [d['step'] for d in test_sub]
    
    _ = plt.plot(controller_step, mean_reward, label=os.path.basename(p) + '_test', alpha=0.75)

_ = plt.legend(loc='lower right')
_ = plt.xlabel('controller_step')
_ = plt.ylabel('mean_reward')

_ = plt.ylim(0.8, 1.0)

for i in np.linspace(0.90, 1.0, 6):
    _ = plt.axhline(i, c='grey', alpha=0.1)

show_plot()
