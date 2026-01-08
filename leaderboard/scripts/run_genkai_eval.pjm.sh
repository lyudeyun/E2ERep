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
# 兼容：即便未来加了 `set -u`，也不因 LD_LIBRARY_PATH 未定义而崩
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# CARLA 启动参数（超算上 Vulkan 容易出问题时，建议用 -opengl）
export CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--opengl}"
# CARLA 启动后等待时间（秒），超算首次启动可能更慢，建议 >= 60
export CARLA_STARTUP_SLEEP="${CARLA_STARTUP_SLEEP:-60}"
# 把 CARLA server 的 stdout/stderr 落盘，方便定位启动失败原因
export CARLA_SERVER_LOG_DIR="${CARLA_SERVER_LOG_DIR:-${PWD}/carla_server_logs}"

# 可选：如果不想保存图片，可以设置 SAVE_PATH 为空（但需要修改脚本逻辑）
# export SAVE_PATH=

# 运行评估脚本
bash leaderboard/scripts/run_evaluation_genkai_vad.sh

