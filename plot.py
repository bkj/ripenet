import sys
import json
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

for f in sys.stdin:
    f = f.strip()
    x = list(map(json.loads, open(f)))
    
    epochs   = [xx['epoch'] for xx in x]
    test_acc = [xx['test_acc'] for xx in x]
    
    _ = plt.plot(epochs, test_acc, alpha=0.25)

_ = plt.grid()
_ = plt.ylim(0.5, 1)
_ = plt.axhline(0.90, lw=1, c='red')
show_plot()