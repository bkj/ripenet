#!/usr/bin/env python

"""
    prep.py
"""

from __future__ import print_function

import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num-nodes', type=int, default=2)   # Number of ops to sample
    parser.add_argument('--num-ops', type=int, default=6) # Number of cells to sample
    parser.add_argument('--population-size', type=int, default=80) 
    
    parser.add_argument('--run', type=str, default='runs/run0')
    parser.add_argument('--num-gpus', type=int, default=4)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


def initialize_population(num_nodes, num_ops, population_size):
    archs = []
    for block_id in range(num_nodes):
        in_left  = np.random.choice(block_id + 1, population_size)
        in_right = np.random.choice(block_id + 1, population_size)
        op_left  = np.random.choice(num_ops, population_size)
        op_right = np.random.choice(num_ops, population_size)
        archs.append(np.column_stack([in_left, in_right, op_left, op_right]))
    
    return np.hstack(archs)



if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.run):
        os.makedirs(args.run)
    
    base_cmd = ['#!/bin/bash\n'] + ['./run-%d.sh &&' % gpu_id for gpu_id in range(args.num_gpus)] + ['\n\n']
    open(os.path.join(args.run, 'run.sh'), 'w').write('\n'.join(base_cmd))
    
    population = initialize_population(
        num_nodes=args.num_nodes,
        num_ops=args.num_ops,
        population_size=args.population_size,
    )
    
    for gpu_id, chunk in enumerate(np.array_split(population, args.num_gpus)):
        file = open(os.path.join(args.run, 'run-%d.sh' % gpu_id), 'w')
        for arch in chunk:
            arch = ''.join(map(str, arch))
            print('CUDA_VISIBLE_DEVICES=%d python train.py --architecture %s --outpath %s' % 
                (gpu_id, arch, os.path.join(args.run, 'results/')), file=file)