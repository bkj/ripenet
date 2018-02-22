
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

# Accuracy of 0.???? (because 1111 just passes identity)