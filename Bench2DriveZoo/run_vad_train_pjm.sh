#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=2
#PJM -L "elapse=100:00:00"
#PJM -j
#PJM -o vad_train_%j.log
#PJM -m e
#PJM --mail-list lyudeyun@gmail.com

# 切换到工作目录
cd /home/pj25001076/ku50002427/git/B2DRepair/Bench2DriveZoo || exit 1

# 激活 conda 环境（按需修改）
source ~/.bashrc
conda activate b2d_zoo

# 训练命令（2 张卡）
./adzoo/vad/dist_train.sh ./adzoo/vad/configs/VAD/VAD_tiny_e2e_b2d.py 2
