#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=2
#PJM -L "elapse=100:00:00"
#PJM -j
#PJM -o vad_train_%j.log
#PJM -m e
#PJM --mail-list you@example.com

# Change to working directory
cd /home/pj25001076/ku50002427/git/B2DRepair/Bench2DriveZoo || exit 1

# Activate conda env (edit as needed)
source ~/.bashrc
conda activate b2d_zoo

# Training (2 GPUs)
./adzoo/vad/dist_train.sh ./adzoo/vad/configs/VAD/VAD_tiny_e2e_b2d.py 2
