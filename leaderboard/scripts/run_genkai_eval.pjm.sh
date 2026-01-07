#!/bin/bash
# 参考你之前的脚本风格，使用 c-batch 队列
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j

# 激活 conda 环境
source /home/pj25001076/ku50002427/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

# 设置工作目录（改成你的实际路径）
cd /home/pj25001076/ku50002427/git/B2DRepair

# 设置 CARLA_ROOT（改成你的实际路径）
export CARLA_ROOT=/home/pj25001076/ku50002427/git/B2DRepair/Bench2DriveZoo/carla

# -------- genkai 环境兼容性设置 --------
# CARLA 的 CarlaUE4.sh 会调用 `xdg-user-dir`，但在某些超算环境里该命令不存在，
# 会导致启动脚本报错/异常。这里提供一个最小的 stub，返回 $HOME，避免阻塞启动。
if ! command -v xdg-user-dir >/dev/null 2>&1; then
  mkdir -p "${HOME}/bin"
  cat > "${HOME}/bin/xdg-user-dir" <<'EOF'
#!/bin/sh
echo "${HOME}"
EOF
  chmod +x "${HOME}/bin/xdg-user-dir"
  export PATH="${HOME}/bin:${PATH}"
fi

# 无显示环境下运行 CARLA（常见于超算/容器）
export DISPLAY=

# 设 libjpeg.so.8（部分环境需要 conda 里的 libjpeg）
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 可选：如果不想保存图片，可以设置 SAVE_PATH 为空（但需要修改脚本逻辑）
# export SAVE_PATH=

# 运行评估脚本
bash leaderboard/scripts/run_evaluation_genkai_vad.sh

