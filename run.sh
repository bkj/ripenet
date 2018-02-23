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


python tests/train_cell_worker.py \
    --lr-schedule constant \
    --lr-init 0.1 \
    --outpath delete-me


mkdir _results/trained
iter=0
python cell-main.py \
    --child child \
    --outpath _results/trained/trained.$iter \
    --child-lr-init 0.1

# Works if children don't control path

# mkdir _results/trained
# iter=0
# python cell-main.py \
#     --dataset cifar10 \
#     --child child \
#     --algorithm ppo \
#     --outpath _results/trained/trained.$iter \
#     --epochs 1000 \
#     --num-ops 3 \
#     --temperature 5.0 \
#     --clip-logits 2.5 \
#     --entropy-penalty 0.1 \
#     --controller-lr 0.00035 \
#     --child-lr-init 0.05 \
#     --child-lr-schedule sgdr \
#     --child-sgdr-period-length 10 \
#     --child-sgdr-t-mult 2

# !! Excluding pooling operations, since they seem to be causing trouble

# ====================================================================
# fashionMNIST + MNIST

# Training on fashionMNIST

mkdir -p _results/fashion
mkdir -p _results/fashion/pretrained

# --
# Pretraining

ARCH="0001" # Identity cell
DATASET="mnist"
python tests/train_cell_worker.py \
    --dataset $DATASET \
    --outpath _results/fashion/pretrained/$DATASET-$ARCH-1x1 \
    --architecture $ARCH \
    --lr-schedule linear \
    --epochs 20 \
    --train-size 1.0 \
    --lr-init 0.01 \
    --num-nodes 1


ARCH="0002_0002" # Two convolutions
DATASET="mnist"
python tests/train_cell_worker.py \
    --dataset $DATASET \
    --outpath _results/fashion/pretrained/$DATASET-$ARCH \
    --architecture $ARCH \
    --lr-schedule constant \
    --epochs 100 \
    --train-size 0.9 \
    --lr-init 0.01 \
    --num-nodes 2




# --
# PPO Training

iter=0
DATASET="mnist"
python cell-main.py \
    --dataset $DATASET \
    --algorithm ppo \
    --outpath _results/fashion/$DATASET-trained.$iter \
    --child child \
    --train-size 0.9 \
    --num-ops 4 \
    --num-nodes 1 \
    --child-lr-init 0.01 \
    --epochs 1000

# Seems to work

iter=9
DATASET="mnist"
python cell-main.py \
    --dataset $DATASET \
    --algorithm ppo \
    --outpath _results/fashion/$DATASET-trained.$iter \
    --child child \
    --train-size 0.9 \
    --num-ops 6 \
    --num-nodes 3 \
    --child-lr-init 0.01 \
    --epochs 1000 \
    --test-topk 10

# !! Tweak MNIST model to give very good performance at baseline
# !! SGDR
# !! FashionMNIST



