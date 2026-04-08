#!/bin/bash
BASE_PORT=30200
BASE_TM_PORT=50200
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=Bench2DriveZoo/team_code/uniad_b2d_agent.py
# TEAM_CONFIG=your_team_agent_ckpt.pth   # for TCP and ADMLP
TEAM_CONFIG=Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+Bench2DriveZoo/ckpts/uniad_base_b2d.pth # for UniAD and VAD
BASE_CHECKPOINT_ENDPOINT=uniad_base_bench2drive220_eval
SAVE_PATH=./uniad_base_bench2drive220/
PLANNER_TYPE=only_traj

# Optional email notification (uses send_email_notification in run_experiment.py via run_evaluation.sh)
# Default OFF; pass --email to enable (email on script exit, e.g. SIGTERM; see run_evaluation.sh).
# Usage:
#   bash leaderboard/scripts/run_evaluation_single_agent.sh                  # no email (default)
#   bash leaderboard/scripts/run_evaluation_single_agent.sh --email        # enable email
#   bash leaderboard/scripts/run_evaluation_single_agent.sh --email=false    # explicit off
NOTIFY_EMAIL=False

for arg in "$@"; do
  case "$arg" in
    --email|--email=true|--email=True)
      NOTIFY_EMAIL=True
      ;;
    --email=false|--email=False)
      NOTIFY_EMAIL=False
      ;;
  esac
done

GPU_RANK=0
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
export NOTIFY_EMAIL
# Save full stdout/stderr next to checkpoint json (same basename, .log suffix)
# Use "%.*" to strip the last extension, avoiding "xxx.json.log".
LOG_FILE="${CHECKPOINT_ENDPOINT%.*}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

bash leaderboard/scripts/run_evaluation.sh \
  $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \
  2>&1 | tee "${LOG_FILE}"

exit ${PIPESTATUS[0]}