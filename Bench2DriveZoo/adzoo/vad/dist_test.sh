#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# For single GPU, use non-distributed mode to avoid ZeroDivisionError
if [ "$GPUS" -eq 1 ]; then
    # Explicitly specify the GPU device
    export CUDA_VISIBLE_DEVICES=0
    python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher none ${@:4} --eval bbox
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
fi

