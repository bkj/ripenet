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
