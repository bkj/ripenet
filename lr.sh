#!/bin/bash

# lr.sh

mkdir -p lr_arches

function train {
    echo "GPU=$CUDA_VISIBLE_DEVICES"
    for ARCH in $@; do
        echo "ARCH=$ARCH"
        python train_cell_worker.py --outpath lr_arches/cell_worker-$ARCH --architecture $ARCH > lr_arches/cell_worker-$ARCH.log    
    done
}
export -f train

CUDA_VISIBLE_DEVICES=0 train 00111000 00321011 00421043 00050131 00241132 00051042 00021041 00130114 00531150 00111105 00241104
CUDA_VISIBLE_DEVICES=1 train 00000154 00000153 00230150 00521044 00441130 00440052 00321015 00120031 00351115 00130101 00130141 00141032


ARCH="00321011"
python train_cell_worker.py --outpath lr_arches/cell_worker-$ARCH --architecture $ARCH > lr_arches/cell_worker-$ARCH.log