
# --
# Noop

mkdir -p _results/ftop
DATASET="mnist"
ARCH="0000"
python tests/train_ftop_worker.py \
    --dataset $DATASET \
    --outpath _results/ftop/$ARCH \
    --lr-schedule cyclical \
    --epochs 30 \
    --train-size 0.9 \
    --lr-init 0.1 \
    --architecture $ARCH

# Accuracy of 0.1 (because 0000 just returns 0s)

# --
# Identity
DATASET="mnist"
ARCH="1111"
python tests/train_ftop_worker.py \
    --dataset $DATASET \
    --outpath _results/ftop/$ARCH \
    --lr-schedule cyclical \
    --epochs 30 \
    --train-size 0.9 \
    --lr-init 0.1 \
    --architecture $ARCH

# Accuracy of 0.984 (because 1111 just passes identity)

# --

DATASET="mnist"
ARCH="2345"
python tests/train_ftop_worker.py \
    --dataset $DATASET \
    --outpath _results/ftop/$ARCH \
    --lr-schedule cyclical \
    --epochs 30 \
    --train-size 0.9 \
    --lr-init 0.1 \
    --architecture $ARCH

# ... should do better because we're learning FTOP ...

DATASET="mnist"
RUN=0
python ftop-main.py \
    --dataset $DATASET \
    --algorithm ppo \
    --outpath _results/ftop/run.$RUN \
    --epochs 300 \
    --train-size 0.9 \
    --child child \
    --child-lr-init 0.01 \
    --test-topk 10
