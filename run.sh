#!/bin/bash

# run.sh

# --
# Testing on a fixed architecture

mkdir -p _results

python main.py --architecture mlp --algorithm reinforce > _results/mlp-reinforce
python main.py --architecture mlp --algorithm ppo > _results/mlp-ppo

python main.py --architecture lstm --algorithm reinforce > _results/lstm-reinforce
python main.py --architecture lstm --algorithm ppo > _results/lstm-ppo

# --
# Train a CellWorker model (should be approximately the same as preactivation ResNet18, though not exactly)

python tests/train_cell_worker.py > pretrained_models/cell_worker-20
python tests/train_cell_worker.py --outpath pretrained_models/cell_worker-50.weights > pretrained_models/cell_worker-50.log

ARCH="00000000"
python tests/train_cell_worker.py --outpath pretrained_models/cell_worker-$ARCH --architecture $ARCH > pretrained_models/cell_worker-$ARCH.log


# --
# Testing cell search

mkdir -p _results/reduced
for iter in $(seq 10); do
    for i in $(seq 3 8); do
        python cell-main.py \
            --algorithm ppo \
            --outpath _results/reduced/cell-ppo.$i.$iter \
            --output-channels $i
    done;
done;

# w/ 7-8 operations, this stops converging w/ the default parameters
# This is either because the search space is too large, or because the pooling layers mess things up

# --
# Training a model w/ cell search

iter=3.1
python cell-main.py \
    --child child \
    --outpath _results/trained/trained.$iter \
    --num-ops 5 \
    --child-lr-init 0.01 \
    --child-lr-schedule sgdr \
    --child-sgdr-period-length 10 \
    --child-sgdr-t-mult 2 \
    --temperature 2.0 \
    --epochs 1000

# !! Should implement some evolutionary method -- could control the convergence issue better probably
