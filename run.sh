#!/bin/bash

# run.sh

# --
# Installation

conda create -n ripenet python=3.5 pip

conda install pytorch torchvision cuda90 -c pytorch
pip install -r requirements.txt

cd ~/projects/basenet
pip install -r requirements.txt
python setup.py clean --all install
cd ~/projects/ripenet

python tests/test.py

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

# mkdir _results/trained
# iter=0
# python cell-main.py \
#     --algorithm ppo \
#     --outpath _results/trained/trained.$iter \
#     --num-ops 6 \
#     --temperature 2.5 \
#     --entropy-penalty 0.1 \
#     --epochs 1000 \
#     --controller-lr 0.00035 \
#     --child child \
#     --pretrained-path pretrained_models/cell_worker-00000000


# iter=1
# python cell-main.py \
#     --algorithm ppo \
#     --outpath _results/trained/trained.$iter \
#     --num-ops 6 \
#     --temperature 2.5 \
#     --epochs 1000 \
#     --controller-lr 0.00035 \
#     --child child

# # 500 epochs, converges to a null cell

# --
# Training on fashionMNIST

iter=0
mkdir _results/fashion
python cell-main.py \
    --algorithm ppo \
    --outpath _results/fashion/trained.$iter \
    --dataset fashion_mnist \
    --num-ops 6 \
    --temperature 2.5 \
    --epochs 1000 \
    --controller-lr 0.00035 \
    --child child
