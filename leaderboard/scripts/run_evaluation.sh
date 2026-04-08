#!/bin/bash
# Repository root (script lives in leaderboard/scripts/); used for default CARLA and PYTHONPATH
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# If CARLA_ROOT is unset, try Bench2DriveZoo-bundled CARLA first
if [ -z "${CARLA_ROOT:-}" ]; then
  DEFAULT_CARLA="${REPO_ROOT}/Bench2DriveZoo/carla"
  if [ -f "${DEFAULT_CARLA}/CarlaUE4.sh" ] && [ -d "${DEFAULT_CARLA}/PythonAPI/carla/agents" ]; then
    export CARLA_ROOT="${DEFAULT_CARLA}"
    echo "INFO: CARLA_ROOT not set, using default: ${CARLA_ROOT}"
  else
    export CARLA_ROOT=YOUR_CARLA_PATH
    echo "WARNING: CARLA_ROOT not set. Set it manually, e.g. export CARLA_ROOT=/path/to/carla" 1>&2
  fi
fi

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# Remove inherited CARLA PythonAPI / egg entries first.
# Runtime evidence: import was crashing inside carla-0.9.15-py3.7...egg while this env uses Python 3.10.
# Keep unrelated PYTHONPATH entries, but ensure only the current CARLA_ROOT paths are added back below.
PYTHONPATH="$(python3 - <<'PY'
import os
parts = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
filtered = []
for p in parts:
    lp = p.lower()
    if "/pythonapi" in lp or ("carla-" in lp and lp.endswith(".egg")):
        continue
    filtered.append(p)
print(":".join(filtered))
PY
)"
export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla

# CARLA .egg ABI must match the Python that runs leaderboard_evaluator.py (not the GPU model).
# Auto-pick: same interpreter as ${PYTHON:-python} (override PYTHON if you use e.g. python3.10).
# Manual override: export CARLA_PY_EGG=/path/to/carla-0.9.15-py3.10-linux-x86_64.egg
if [ -n "${CARLA_PY_EGG:-}" ] && [ -f "$CARLA_PY_EGG" ]; then
  export PYTHONPATH=$PYTHONPATH:$CARLA_PY_EGG
else
  _PY="${PYTHON:-python}"
  _MM="$(command -v "$_PY" >/dev/null 2>&1 && "$_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  _MM="${_MM:-10}"
  _DIST="$CARLA_ROOT/PythonAPI/carla/dist"
  _TRY="$_DIST/carla-0.9.15-py${_MM}-linux-x86_64.egg"
  if [ -f "$_TRY" ]; then
    export PYTHONPATH=$PYTHONPATH:$_TRY
  else
    _FIRST="$(ls "$_DIST"/carla-*-py3.*-linux-x86_64.egg 2>/dev/null | head -1)"
    if [ -n "$_FIRST" ] && [ -f "$_FIRST" ]; then
      export PYTHONPATH=$PYTHONPATH:$_FIRST
    else
      echo "WARNING: No CARLA egg in $_DIST for Python ${_MM}; set CARLA_PY_EGG. Expected: $_TRY" >&2
      export PYTHONPATH=$PYTHONPATH:$_TRY
    fi
  fi
fi

# Leaderboard, scenario_runner, Bench2DriveZoo (absolute paths so cwd does not matter)
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/leaderboard"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/leaderboard/team_code"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/scenario_runner"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/Bench2DriveZoo"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/Bench2DriveZoo/team_code"
export SCENARIO_RUNNER_ROOT="${REPO_ROOT}/scenario_runner"

export LEADERBOARD_ROOT="${REPO_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

EVAL_EMAIL_SENT=0

# Optional email: also run on EXIT so we still notify when the child exits with SIGTERM (e.g. 143) or the shell is interrupted.
send_eval_email_once() {
  if [ "${EVAL_EMAIL_SENT:-0}" = "1" ]; then
    return 0
  fi
  if [ "${NOTIFY_EMAIL:-False}" != "True" ] && [ "${NOTIFY_EMAIL:-False}" != "true" ] && [ "${NOTIFY_EMAIL:-0}" != "1" ]; then
    return 0
  fi
  EVAL_EMAIL_SENT=1
  export PIPE_EXIT_CODE="${PIPE_EXIT_CODE:-$?}"
  python - <<'PY'
import os
from datetime import datetime

subject_prefix = os.environ.get("EMAIL_SUBJECT_PREFIX", "[B2DRepair]")
exit_code = os.environ.get("PIPE_EXIT_CODE", "")


def decode_exit(ec: str):
    try:
        n = int(ec)
    except Exception:
        return ec, ""
    if 128 < n < 256:
        sig = n - 128
        names = {1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 9: "SIGKILL", 15: "SIGTERM"}
        name = names.get(sig, f"signal {sig}")
        return ec, f" (stopped by {name}, 128+{sig})"
    return ec, ""


ec_str, sig_hint = decode_exit(exit_code)

try:
    from run_experiment import send_email_notification
except Exception as e:
    print(f"[EMAIL] import send_email_notification failed: {e}")
else:
    ckpt = os.environ.get("CHECKPOINT_ENDPOINT", "")
    save_path = os.environ.get("SAVE_PATH", "")
    routes = os.environ.get("ROUTES", "")
    agent = os.environ.get("TEAM_AGENT", "")
    port = os.environ.get("PORT", "")
    tm_port = os.environ.get("TM_PORT", "")
    gpu_rank = os.environ.get("GPU_RANK", "")
    status = "FINISHED" if exit_code == "0" else "FAILED"
    subj = f"{subject_prefix} closed-loop eval {status}"
    if "SIGTERM" in sig_hint or "(128+15)" in sig_hint:
        subj = f"{subject_prefix} closed-loop eval {status} (SIGTERM/Terminated)"
    elif "SIGKILL" in sig_hint or "(128+9)" in sig_hint:
        subj = f"{subject_prefix} closed-loop eval {status} (SIGKILL)"
    body = "\n".join([
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Status: {status}",
        f"Exit code: {ec_str}{sig_hint}",
        f"Routes: {routes}",
        f"Checkpoint: {ckpt}",
        f"Save path: {save_path}",
        f"Agent: {agent}",
        f"Port: {port}",
        f"TM port: {tm_port}",
        f"GPU rank: {gpu_rank}",
        "",
        "Note: exit code 143 often means the Python process received SIGTERM (Terminated).",
        "SIGKILL (137) cannot be intercepted; email is sent only if the shell exits cleanly afterward.",
    ])
    send_email_notification(subj, body)
PY
}

trap 'send_eval_email_once' EXIT

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK}

exit_code=$?
export PIPE_EXIT_CODE=$exit_code

exit "$exit_code"
