#!/bin/bash

# hyperband.sh

# --
# Train model w/ known good architecture

ARCH="0002_0112"
python train_cell_worker.py \
    --outpath _results/hyperband/pretrained-$ARCH \
    --architecture $ARCH \
    --lr-schedule linear \
    --lr-init 0.1 \
    --epochs 20 \
    --train-size 1.0 > _results/hyperband/pretrained-$ARCH.log

# {"epoch": 18, "train_acc": 0.96568, "val_acc": null, "test_acc": 0.9157}
# {"epoch": 19, "train_acc": 0.97652, "val_acc": null, "test_acc": 0.9232}

# --
# Train model w/ suboptimal learning rate schedule

ARCH="0002_0112"
python train_cell_worker.py \
    --outpath _results/hyperband/pretrained-constant-$ARCH \
    --architecture $ARCH \
    --lr-schedule constant \
    --lr-init 0.01 \
    --epochs 50 \
    --train-size 0.9 > _results/hyperband/pretrained-constant-$ARCH.log

# {"epoch": 48, "train_acc": 0.9708444444444444, "val_acc": 0.8856, "test_acc": 0.8933}
# {"epoch": 49, "train_acc": 0.9740222222222222, "val_acc": 0.902, "test_acc": 0.9079}

# --
# How does changing the balance of good/bad architectures effect the good architecture?

mkdir -p _results/hyperband
iter=12
python cell-main.py \
    --algorithm hyperband \
    --outpath _results/hyperband/hyperband.$iter \
    --controller-eval-interval 1 \
    --child-lr-init 0.01 \
    --epochs 200 \
    --num-ops 4 \
    --seed 678 \
    --train-size 0.9

# _results/hyperband/hyperband.1.log -> 7/8 0002_0112/0011_1130
# _results/hyperband/hyperband.2.log -> 6/8 0002_0112/0011_1130
# _results/hyperband/hyperband.3.log -> 5/8 0002_0112/0011_1130

# _results/hyperband/hyperband.4.log -> 6/8 0002_0112/0011_1130, pushed weights into cells # No shared weights!
# _results/hyperband/hyperband.5.log -> 4/4 0002_0112/0011_1130, pushed weights into cells # No shared weights!

# _results/hyperband/hyperband.6.log -> 6/8 0002_0112/0012_0112

# _results/hyperband/hyperband.{7,8,9}.log ->  (different seeds, NOOP was actually identity in 7 and 8)
# [[0 0 0 2 0 1 1 2]
# [0 0 1 2 0 1 1 2]
# [0 0 2 2 0 1 1 2]
# [0 0 3 2 0 1 1 2]
# [0 0 0 3 0 1 1 2]
# [0 0 1 3 0 1 1 2]
# [0 0 2 3 0 1 1 2]
# [0 0 3 3 0 1 1 2]]

# _results/hyperband/hyperband.{10,11,12}.log -> (11 does longer eval epochs, 12 fixes BN)
 # [[0 0 0 2 0 1 1 2]
 # [0 0 1 2 0 1 1 2]
 # [0 0 2 2 0 1 1 2]
 # [0 0 3 2 0 1 1 2]
 # [0 0 0 3 0 1 1 2]
 # [0 0 1 3 0 1 1 2]
 # [0 0 2 3 0 1 1 2]
 # [0 0 3 3 0 1 1 2]
 # [0 0 0 2 0 1 2 2]
 # [0 0 1 2 0 1 2 2]
 # [0 0 2 2 0 1 2 2]
 # [0 0 3 2 0 1 2 2]
 # [0 0 0 3 0 1 3 2]
 # [0 0 1 3 0 1 3 2]
 # [0 0 2 3 0 1 3 2]
 # [0 0 3 3 0 1 3 2]]

# --
# Training architectures individualls

ARCH="0002_0112"
python train_cell_worker.py \
    --outpath _results/hyperband/pretrained-constant-$ARCH \
    --architecture $ARCH \
    --lr-schedule constant \
    --lr-init 0.01 \
    --epochs 50 \
    --train-size 0.9 > _results/hyperband/pretrained-constant-$ARCH.log

for ARCH in $(cat arches); do
    echo $ARCH
    python train_cell_worker.py \
        --outpath _results/hyperband/pretrained-constant-$ARCH \
        --architecture $ARCH \
        --lr-schedule constant \
        --lr-init 0.01 \
        --epochs 50 \
        --train-size 0.9 > _results/hyperband/pretrained-constant-$ARCH.log
done