#!/usr/bin/env python

"""
    downselect.py
"""

from __future__ import print_function

import os
import sys
import json
import argparse
import numpy as np
from glob import glob

CLEANUP = (
'cleanup() {'
'    echo "cleanup";'
'    local pids=$(jobs -pr);'
'    [ -n "$pids" ] && kill $pids;'
'};\n'
'trap "cleanup" INT QUIT TERM EXIT'
'\n\n'
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--population-size', type=int, default=40)
    parser.add_argument('--run', type=str, default='runs/delete-me')
    parser.add_argument('--num-gpus', type=int, default=5)
    
    return parser.parse_args()


def load_population(population_size):
    population = {}
    
    for f in sys.stdin:
        f = f.strip()
        arch = os.path.basename(f).split('_')[0]
        hist = list(map(json.loads, open(f)))
        if len(hist) > 0:
            test_acc = hist[-1]['test_acc']
            population[arch] = test_acc
            
    population = sorted(population.items(), key=lambda x: -x[1])
    population = population[:population_size]
    
    return [p[0] for p in population]


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.run):
        os.makedirs(args.run)
    
    base_cmd = ['#!/bin/bash\n', CLEANUP] + ['./run-%d.sh &' % gpu_id for gpu_id in range(args.num_gpus)] + ['wait; echo done\n\n']
    open(os.path.join(args.run, 'run.sh'), 'w').write('\n'.join(base_cmd))
    
    population = load_population(population_size=args.population_size)
    
    for gpu_id, chunk in enumerate(np.array_split(population, args.num_gpus)):
        file = open(os.path.join(args.run, 'run-%d.sh' % gpu_id), 'w')
        for arch in chunk:
            print('CUDA_VISIBLE_DEVICES=%d python %s/train.py --architecture %s --outpath %s --epochs %d' % 
                (gpu_id, os.getcwd(), arch, os.path.join(os.getcwd(), args.run, 'results/'), args.epochs), file=file)