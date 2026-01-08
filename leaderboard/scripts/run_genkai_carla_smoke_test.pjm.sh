#!/bin/bash
# Genkai CARLA Smoke Test (PJM)
# 目标：用一次很短的作业，验证 CARLA 能否在该节点启动并接受 carla.Client 连接。
#
# 用法：
#   pjsub leaderboard/scripts/run_genkai_carla_smoke_test.pjm.sh
#
# 产物：
#   - ${PWD}/carla_server_logs/smoke_carla_<PORT>.log  (CARLA server stdout/stderr)
#
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=00:15:00
#PJM -j

set -euxo pipefail

# -------------------- conda 环境 --------------------
source /home/pj25001076/ku50002427/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

# -------------------- 工作目录（按需修改） --------------------
cd /home/pj25001076/ku50002427/git/B2DRepair

# -------------------- CARLA_ROOT（按需修改） --------------------
export CARLA_ROOT=/home/pj25001076/ku50002427/git/B2DRepair/Bench2DriveZoo/carla

# -------- genkai 环境兼容性设置 --------
# CarlaUE4.sh 有时会调用 `xdg-user-dir`，超算环境可能不存在；提供最小 stub。
if ! command -v xdg-user-dir >/dev/null 2>&1; then
  mkdir -p "${HOME}/bin"
  cat > "${HOME}/bin/xdg-user-dir" <<'EOF'
#!/bin/sh
echo "${HOME}"
EOF
  chmod +x "${HOME}/bin/xdg-user-dir"
  export PATH="${HOME}/bin:${PATH}"
fi

# 无显示环境下运行 CARLA
export DISPLAY=

# 部分环境需要 conda 里的 libjpeg/libstdc++ 等
# 注意：脚本使用了 `set -u`，当 LD_LIBRARY_PATH 未定义时直接引用会报 “未割り当ての変数”
# 用安全的参数展开：若原变量存在则追加，否则只设置 conda 的 lib
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 建议先用 -opengl 绕开 Vulkan 相关问题；如需改回 Vulkan：export CARLA_EXTRA_ARGS=""
export CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--opengl}"

# CARLA 启动后等待时间（秒）：节点慢的话可以改大，比如 120
export CARLA_STARTUP_SLEEP="${CARLA_STARTUP_SLEEP:-60}"

# 日志目录
LOGDIR="${PWD}/carla_server_logs"
mkdir -p "${LOGDIR}"

# 找一个可用端口（默认从 30000 开始）
PORT_BASE="${PORT_BASE:-30000}"
PORT="$(python - <<PY
import socket
port=int(${PORT_BASE})
while True:
    try:
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("localhost", port))
        s.close()
        print(port)
        break
    except OSError:
        port += 1
PY
)"

SERVER_LOG="${LOGDIR}/smoke_carla_${PORT}.log"

echo "=================================================="
echo "CARLA smoke test"
echo "  CARLA_ROOT=${CARLA_ROOT}"
echo "  PORT=${PORT}"
echo "  CARLA_EXTRA_ARGS=${CARLA_EXTRA_ARGS}"
echo "  CARLA_STARTUP_SLEEP=${CARLA_STARTUP_SLEEP}"
echo "  SERVER_LOG=${SERVER_LOG}"
echo "=================================================="

# 启动 CARLA server（用 setsid，方便 kill 整个进程组）
setsid "${CARLA_ROOT}/CarlaUE4.sh" -RenderOffScreen -nosound \
  -carla-rpc-port="${PORT}" -graphicsadapter=0 ${CARLA_EXTRA_ARGS} \
  > "${SERVER_LOG}" 2>&1 &
CARLA_PID=$!

cleanup() {
  set +e
  if [ -n "${CARLA_PID:-}" ]; then
    # kill process group
    kill -- -"${CARLA_PID}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 -- -"${CARLA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep "${CARLA_STARTUP_SLEEP}"

# 连接测试：能 get_world 就算 smoke test 通过
python - <<PY
import carla
client = carla.Client("localhost", ${PORT})
client.set_timeout(10.0)
world = client.get_world()
print("OK: connected. map =", world.get_map().name)
PY

echo "SMOKE TEST PASSED"


