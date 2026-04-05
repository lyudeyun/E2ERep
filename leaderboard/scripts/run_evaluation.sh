#!/bin/bash
# Repository root (used for default CARLA path and PYTHONPATH)
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
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

# Leaderboard & agents
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/leaderboard"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/leaderboard/team_code"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/scenario_runner"

# Bench2DriveZoo UniAD/VAD agents
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/Bench2DriveZoo"
export PYTHONPATH=$PYTHONPATH:"${REPO_ROOT}/Bench2DriveZoo/team_code"
export SCENARIO_RUNNER_ROOT=scenario_runner

export LEADERBOARD_ROOT=leaderboard
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
--gpu-rank=${GPU_RANK} \
