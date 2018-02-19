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

# --
# 

python cell-main.py --pretrained-path pretrained_models/cell_worker-50.weights