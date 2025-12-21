#!/bin/bash
# Closed-loop evaluation entrypoint for Bench2Drive/Leaderboard.
#
# This script is meant to be called by run_experiment.py (scheme B).
# It is parameterized (unlike run_evaluation_debug.sh) and does NOT override CARLA_ROOT.
#
# Usage:
#   bash leaderboard/scripts/run_closed_loop_eval.sh \
#     PORT TM_PORT IS_BENCH2DRIVE ROUTES_XML TEAM_AGENT TEAM_CONFIG CHECKPOINT_JSON SAVE_PATH PLANNER_TYPE GPU_RANK
#
# Requirements:
#   - Run inside your b2d_zoo conda env (so python deps are available)
#   - Export CARLA_ROOT=/path/to/CARLA
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash leaderboard/scripts/run_closed_loop_eval.sh \
    PORT TM_PORT IS_BENCH2DRIVE ROUTES_XML TEAM_AGENT TEAM_CONFIG CHECKPOINT_JSON SAVE_PATH PLANNER_TYPE GPU_RANK

Example:
  export CARLA_ROOT=/data/carla-0.9.15
  bash leaderboard/scripts/run_closed_loop_eval.sh 30000 50000 True \
    leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    Bench2DriveZoo/team_code/vad_b2d_agent.py \
    Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+<ckpt.pth> \
    /tmp/closed_loop_eval.json /tmp/closed_loop_eval only_traj 0
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -ne 10 ]; then
  echo "ERROR: Expected 10 arguments, got $#." 1>&2
  usage
  exit 2
fi

PORT="$1"
TM_PORT="$2"
IS_BENCH2DRIVE="$3"
ROUTES="$4"
TEAM_AGENT="$5"
TEAM_CONFIG="$6"
CHECKPOINT_ENDPOINT="$7"
SAVE_PATH="$8"
PLANNER_TYPE="$9"
GPU_RANK="${10}"

if [ -z "${CARLA_ROOT:-}" ]; then
  echo "ERROR: CARLA_ROOT is not set. Please export CARLA_ROOT=/path/to/CARLA" 1>&2
  exit 1
fi

# If user accidentally set "home/..." (missing leading slash), normalize it.
if [[ "${CARLA_ROOT}" != /* ]]; then
  CARLA_ROOT="/${CARLA_ROOT}"
fi

# CARLA_ROOT must point to a full CARLA install that contains PythonAPI/carla/agents.
if [ ! -d "${CARLA_ROOT}/PythonAPI/carla" ]; then
  echo "ERROR: ${CARLA_ROOT}/PythonAPI/carla not found." 1>&2
  echo "  Your CARLA_ROOT is probably wrong (it must be the CARLA installation root)." 1>&2
  echo "  Expected to find: ${CARLA_ROOT}/CarlaUE4.sh and ${CARLA_ROOT}/PythonAPI/carla/agents/ ..." 1>&2
  exit 1
fi
if [ ! -d "${CARLA_ROOT}/PythonAPI/carla/agents" ]; then
  echo "ERROR: ${CARLA_ROOT}/PythonAPI/carla/agents not found (required for 'import agents.*')." 1>&2
  echo "  Fix: set CARLA_ROOT to a CARLA install that includes the PythonAPI source tree (not only the .egg)." 1>&2
  exit 1
fi

mkdir -p "$(dirname "${CHECKPOINT_ENDPOINT}")"
mkdir -p "${SAVE_PATH}"

export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI"
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla"
# Keep original egg path for compatibility (if present)
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS

# Provide these to the agent (used for saving frames)
export ROUTES="${ROUTES}"
export SAVE_PATH="${SAVE_PATH}"
export IS_BENCH2DRIVE="${IS_BENCH2DRIVE}"
export PLANNER_TYPE="${PLANNER_TYPE}"

echo "PORT=${PORT}"
echo "TM_PORT=${TM_PORT}"
echo "ROUTES=${ROUTES}"
echo "TEAM_AGENT=${TEAM_AGENT}"
echo "TEAM_CONFIG=${TEAM_CONFIG}"
echo "CHECKPOINT=${CHECKPOINT_ENDPOINT}"
echo "SAVE_PATH=${SAVE_PATH}"
echo "PLANNER_TYPE=${PLANNER_TYPE}"
echo "GPU_RANK=${GPU_RANK}"

CUDA_VISIBLE_DEVICES="${GPU_RANK}" python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --repetitions=1 \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --debug-checkpoint="${SAVE_PATH}/live_results.txt" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug=0 \
  --record="${RECORD_PATH:-}" \
  --resume=True \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}"


