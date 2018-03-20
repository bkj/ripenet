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
    try:
        data = list(map(json.loads, open(p).readlines()))
        mean_reward_good = [d['test_acc'] for d in data]
        epoch = [d['epoch'] for d in data]
        _ = plt.plot(epoch[-30:], mean_reward_good[-30:], label=p, alpha=0.25)#  0.75 * (p == paths[-1]))
    except:
        pass

_ = plt.xlabel('epoch')
_ = plt.ylabel('mean_reward')
# _ = plt.ylim(0.6, 1.0)
_ = plt.grid(c='grey', alpha=0.25)
show_plot() 