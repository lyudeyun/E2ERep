#!/bin/bash
# Set CARLA_ROOT if not already set
if [ -z "${CARLA_ROOT:-}" ]; then
  export CARLA_ROOT="$(pwd)/Bench2DriveZoo/carla"
fi

# Fix CARLA_ROOT if missing leading slash (e.g., "home/..." -> "/home/...")
if [[ "${CARLA_ROOT}" != /* ]]; then
  export CARLA_ROOT="/${CARLA_ROOT}"
fi

IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10
PLANNER_TYPE=only_traj
GPU_RANK=0

# ============================================================================
# Agent and Config Selection
# ============================================================================
# To use super-fast version (640x360, no JPEG encode/decode, no 0.8 scaling):
#   Set USE_SUPER_FAST_VERSION=1
# To use fast version (1280x720, no JPEG encode/decode, no 0.8 scaling):
#   Set USE_FAST_VERSION=1
# To use normal version (default):
#   Set both to 0 or leave them unset
USE_FAST_VERSION="${USE_FAST_VERSION:-0}"
USE_SUPER_FAST_VERSION="${USE_SUPER_FAST_VERSION:-0}"

if [ "${USE_SUPER_FAST_VERSION}" = "1" ]; then
  if [ "${USE_FAST_VERSION}" = "1" ]; then
    echo "ERROR: USE_FAST_VERSION and USE_SUPER_FAST_VERSION are mutually exclusive"
    exit 1
  fi
  VERSION_NAME="super_fast"
  VERSION_DESC="SUPER-FAST (640x360)"
  BASE_PORT=30200
  BASE_TM_PORT=50200
  BASE_CHECKPOINT_ENDPOINT=eval_super_fast
  BASE_SAVE_PATH=./eval_super_fast/
  TEAM_AGENT=Bench2DriveZoo/team_code/vad_b2d_agent_super_fast.py
  TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d_super_fast.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth
elif [ "${USE_FAST_VERSION}" = "1" ]; then
  VERSION_NAME="fast"
  VERSION_DESC="FAST (1280x720)"
  BASE_PORT=30100
  BASE_TM_PORT=50100
  BASE_CHECKPOINT_ENDPOINT=eval_fast
  BASE_SAVE_PATH=./eval_fast/
  TEAM_AGENT=Bench2DriveZoo/team_code/vad_b2d_agent_fast.py
  TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d_fast.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth
else
  VERSION_NAME="normal"
  VERSION_DESC="NORMAL (1600x900)"
  BASE_PORT=30000
  BASE_TM_PORT=50000
  BASE_CHECKPOINT_ENDPOINT=eval
  BASE_SAVE_PATH=./eval_v1/
  TEAM_AGENT=leaderboard/team_code/vad_b2d_agent.py
  TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth
fi

echo "Using ${VERSION_DESC} version agent and config"
echo "  Port: ${BASE_PORT}"
echo "  TM Port: ${BASE_TM_PORT}"
echo "  Checkpoint: ${BASE_CHECKPOINT_ENDPOINT}"
echo "  Save Path: ${BASE_SAVE_PATH}"

PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
SAVE_PATH="${BASE_SAVE_PATH}"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
