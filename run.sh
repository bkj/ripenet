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
# Usage

mkdir -p _results

python main.py --architecture mlp --algorithm reinforce > _results/mlp-reinforce
python main.py --architecture mlp --algorithm ppo > _results/mlp-ppo

python main.py --architecture lstm --algorithm reinforce > _results/lstm-reinforce
python main.py --architecture lstm --algorithm ppo > _results/lstm-ppo