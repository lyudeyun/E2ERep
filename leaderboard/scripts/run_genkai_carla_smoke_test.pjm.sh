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

# 建议先用 -opengl 绕开 Vulkan 相关问题；
# 另外加上 -log/-stdout 尽量把 UE4 日志打到 stdout/stderr，方便我们在 SERVER_LOG 里直接定位问题。
# 如需改回 Vulkan：export CARLA_EXTRA_ARGS=""
export CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--opengl -log -stdout}"

# CARLA 启动后等待时间（秒）：节点慢的话可以改大，比如 120
export CARLA_STARTUP_SLEEP="${CARLA_STARTUP_SLEEP:-60}"
# 连接等待（秒）：有些节点第一次启动（编译 shader/加载资源）会明显更慢
export CARLA_READY_TIMEOUT="${CARLA_READY_TIMEOUT:-180}"
# carla.Client 超时时间（秒）
export CARLA_CLIENT_TIMEOUT="${CARLA_CLIENT_TIMEOUT:-60}"

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

echo "[SMOKE] CARLA_PID=${CARLA_PID}"
ps -p "${CARLA_PID}" -o pid,ppid,stat,etime,cmd || true
pgrep -a "CarlaUE4-Linux-Shipping" || true

sleep "${CARLA_STARTUP_SLEEP}"

# 等待端口就绪 + 进程仍存活（比盲等更可靠）
echo "[SMOKE] waiting for CARLA to listen on localhost:${PORT} (timeout=${CARLA_READY_TIMEOUT}s)"
set +e
python - <<PY
import os, socket, sys, time
port = int(${PORT})
timeout = int(os.environ.get("CARLA_READY_TIMEOUT", "180"))
pid = int(${CARLA_PID})

def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

start = time.time()
while True:
    if not is_pid_alive(pid):
        print(f"[SMOKE][FAIL] CARLA process exited early (pid={pid}).")
        sys.exit(2)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        print("[SMOKE] port is open.")
        sys.exit(0)
    except Exception:
        s.close()
        if time.time() - start > timeout:
            print(f"[SMOKE][FAIL] port {port} not open within {timeout}s.")
            sys.exit(3)
        time.sleep(2)
PY
RC=$?
set -e

if [ "${RC}" -ne 0 ]; then
  echo "[SMOKE] =================================================="
  echo "[SMOKE][FAIL] CARLA not ready (rc=${RC}). Dump diagnostics:"
  echo "[SMOKE]   CARLA_PID=${CARLA_PID}"
  echo "[SMOKE]   SERVER_LOG=${SERVER_LOG}"
  echo "[SMOKE] =================================================="
  ps -p "${CARLA_PID}" -o pid,ppid,stat,etime,cmd || true
  pgrep -a "CarlaUE4-Linux-Shipping" || true
  echo "[SMOKE] server log tail (for quick diagnosis):"
  tail -n 400 "${SERVER_LOG}" || true
  echo "[SMOKE] probing UE4 Saved/Logs (if any):"
  UE4_LOG_DIR="${CARLA_ROOT}/CarlaUE4/Saved/Logs"
  if [ -d "${UE4_LOG_DIR}" ]; then
    ls -lh "${UE4_LOG_DIR}" || true
    tail -n 200 "${UE4_LOG_DIR}"/*.log 2>/dev/null || true
  else
    echo "[SMOKE] ${UE4_LOG_DIR} does not exist"
  fi
  exit "${RC}"
fi

echo "[SMOKE] server log tail (for quick diagnosis):"
tail -n 200 "${SERVER_LOG}" || true
echo "[SMOKE] probing UE4 Saved/Logs (if any):"
UE4_LOG_DIR="${CARLA_ROOT}/CarlaUE4/Saved/Logs"
if [ -d "${UE4_LOG_DIR}" ]; then
  ls -lh "${UE4_LOG_DIR}" || true
  tail -n 200 "${UE4_LOG_DIR}"/*.log 2>/dev/null || true
else
  echo "[SMOKE] ${UE4_LOG_DIR} does not exist"
fi

# 连接测试：能 get_world 就算 smoke test 通过
python - <<PY
import carla
client = carla.Client("localhost", ${PORT})
client.set_timeout(float(${CARLA_CLIENT_TIMEOUT}))
world = client.get_world()
print("OK: connected. map =", world.get_map().name)
PY

echo "SMOKE TEST PASSED"


