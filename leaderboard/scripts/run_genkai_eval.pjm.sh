#!/bin/bash
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=dvall"
#PJM -L "node=1"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=1"
#PJM -j
#PJM -s
#PJM -o "genkai_eval_%j.out"
#PJM -e "genkai_eval_%j.err"

# 激活 conda 环境
source /home/pj25001076/ku50002427/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

# 设置工作目录（改成你的实际路径）
cd /home/pj25001076/ku50002427/git/B2DRepair

# 设置 CARLA_ROOT（改成你的实际路径）
export CARLA_ROOT=/path/to/your/carla

# 可选：如果不想保存图片，可以设置 SAVE_PATH 为空（但需要修改脚本逻辑）
# export SAVE_PATH=

# 运行评估脚本
bash leaderboard/scripts/run_evaluation_genkai_vad.sh

