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

source activate py35

python prep.py --run runs/run_2nodes --num-nodes 2
python prep.py --run runs/run_3nodes --num-nodes 3

cat runs/run_2nodes/run.sh


cd ./runs/run_2nodes/
chmod +x *sh
./run.sh
cd ../../

cd ./runs/run_3nodes/
chmod +x *sh
./run.sh
cd ../../

find runs/run_2nodes | fgrep .log | python plot.py
find runs/run_3nodes | fgrep .log | python plot.py

# --
# Iteration 2

find runs/run_3nodes | fgrep .log | python downselect.py --run runs/run_3nodes_it1 --epochs 30 --population-size 30
cd ./runs/run_3nodes_it1/
chmod +x *sh
./run.sh
cd ../../

# Iteration 3
find runs/run_3nodes_it1 | fgrep .log | python downselect.py --run runs/run_3nodes_it2 --epochs 70 --population-size 15
cd ./runs/run_3nodes_it2/
chmod +x *sh
./run.sh