#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L "elapse=24:00:00"
#PJM -j
#PJM -o repair_job_%j.log
#PJM -m e
# Replace with your address for job-end mail from the scheduler:
#PJM --mail-list recipient@example.com

set -euo pipefail

# ==== Environment (enable module load if needed) ====
# module load matlab
# module load cuda

# ==== Conda ====
source /home/pj25001076/ku50002427/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ==== Repo root ====
cd /home/pj25001076/ku50002427/git/E2ERep || exit 1

# ==== Logs ====
mkdir -p logs

# ==== Runs (scheme A: multiple experiments on one GPU) ====
CUDA_VISIBLE_DEVICES=0 \
python3 run_experiment.py \
  --model-type VAD \
  --model-name VAD_base \
  --rep-method Arachne_v2 \
  --search-algo DE \
  --run-idx 1 \
  --fitness discrete \
  --num-runs 5 \
  --repair-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl \
  --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
  --repair-alpha 0.5 \
  --repair-layers "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2" \
  --repair-num-weights 26 \
  --repair-particles-multiplier 2 \
  --repair-num-iterations 50 \
  --repair-early-stop-patience 5 \
  --eval-cuda-device 0 \
  --regenerate-baseline True \
  --closed-loop-eval False \
  --closed-loop-routes leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --closed-loop-team-agent Bench2DriveZoo/team_code/vad_b2d_agent.py \
  --closed-loop-port 30000 \
  --closed-loop-tm-port 50000 \
  --closed-loop-gpu-rank 0 \
  --closed-loop-planner-type only_traj \
  --closed-loop-is-bench2drive True \
  --time-horizon 3 \
  --occ-output-dir baseline/VAD/vad_occ_cache \
  --collision-num-workers 13 \
  > logs/run_openloop_repair_1.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 run_experiment.py \
  --model-type VAD \
  --model-name VAD_base \
  --rep-method Arachne_v2 \
  --search-algo DE \
  --run-idx 1 \
  --fitness continuous \
  --num-runs 5 \
  --repair-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl \
  --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
  --repair-alpha 0.5 \
  --repair-layers "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2" \
  --repair-num-weights 26 \
  --repair-particles-multiplier 2 \
  --repair-num-iterations 50 \
  --repair-early-stop-patience 5 \
  --eval-cuda-device 0 \
  --regenerate-baseline True \
  --closed-loop-eval False \
  --closed-loop-routes leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --closed-loop-team-agent Bench2DriveZoo/team_code/vad_b2d_agent.py \
  --closed-loop-port 30000 \
  --closed-loop-tm-port 50000 \
  --closed-loop-gpu-rank 0 \
  --closed-loop-planner-type only_traj \
  --closed-loop-is-bench2drive True \
  --time-horizon 3 \
  --occ-output-dir baseline/VAD/vad_occ_cache \
  --collision-num-workers 13 \
  > logs/run_openloop_repair_2.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 run_experiment.py \
  --model-type VAD \
  --model-name VAD_base \
  --rep-method Arachne_v2 \
  --search-algo DE \
  --run-idx 1 \
  --fitness continuous2 \
  --num-runs 5 \
  --repair-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl \
  --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
  --repair-alpha 0.5 \
  --repair-layers "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2" \
  --repair-num-weights 26 \
  --repair-particles-multiplier 2 \
  --repair-num-iterations 50 \
  --repair-early-stop-patience 5 \
  --eval-cuda-device 0 \
  --regenerate-baseline True \
  --closed-loop-eval False \
  --closed-loop-routes leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --closed-loop-team-agent Bench2DriveZoo/team_code/vad_b2d_agent.py \
  --closed-loop-port 30000 \
  --closed-loop-tm-port 50000 \
  --closed-loop-gpu-rank 0 \
  --closed-loop-planner-type only_traj \
  --closed-loop-is-bench2drive True \
  --time-horizon 3 \
  --occ-output-dir baseline/VAD/vad_occ_cache \
  --collision-num-workers 13 \
  > logs/run_openloop_repair_3.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 run_experiment.py \
  --model-type VAD \
  --model-name VAD_base \
  --rep-method semSegRep \
  --search-algo DE \
  --run-idx 1 \
  --fitness discrete \
  --num-runs 5 \
  --repair-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl \
  --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
  --repair-alpha 0.5 \
  --repair-layers "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2" \
  --repair-num-weights 26 \
  --repair-particles-multiplier 2 \
  --repair-num-iterations 50 \
  --repair-early-stop-patience 5 \
  --eval-cuda-device 0 \
  --regenerate-baseline True \
  --closed-loop-eval False \
  --closed-loop-routes leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --closed-loop-team-agent Bench2DriveZoo/team_code/vad_b2d_agent.py \
  --closed-loop-port 30000 \
  --closed-loop-tm-port 50000 \
  --closed-loop-gpu-rank 0 \
  --closed-loop-planner-type only_traj \
  --closed-loop-is-bench2drive True \
  --time-horizon 3 \
  --occ-output-dir baseline/VAD/vad_occ_cache \
  --collision-num-workers 13 \
  > logs/run_openloop_repair_4.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 run_experiment.py \
  --model-type VAD \
  --model-name VAD_base \
  --rep-method semSegRep \
  --search-algo DE \
  --run-idx 1 \
  --fitness continuous \
  --num-runs 5 \
  --repair-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl \
  --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
  --repair-alpha 0.5 \
  --repair-layers "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2" \
  --repair-num-weights 26 \
  --repair-particles-multiplier 2 \
  --repair-num-iterations 50 \
  --repair-early-stop-patience 5 \
  --eval-cuda-device 0 \
  --regenerate-baseline True \
  --closed-loop-eval False \
  --closed-loop-routes leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --closed-loop-team-agent Bench2DriveZoo/team_code/vad_b2d_agent.py \
  --closed-loop-port 30000 \
  --closed-loop-tm-port 50000 \
  --closed-loop-gpu-rank 0 \
  --closed-loop-planner-type only_traj \
  --closed-loop-is-bench2drive True \
  --time-horizon 3 \
  --occ-output-dir baseline/VAD/vad_occ_cache \
  --collision-num-workers 13 \
  > logs/run_openloop_repair_5.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 run_experiment.py \
  --model-type VAD \
  --model-name VAD_base \
  --rep-method semSegRep \
  --search-algo DE \
  --run-idx 1 \
  --fitness continuous2 \
  --num-runs 5 \
  --repair-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl \
  --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
  --repair-alpha 0.5 \
  --repair-layers "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2" \
  --repair-num-weights 26 \
  --repair-particles-multiplier 2 \
  --repair-num-iterations 50 \
  --repair-early-stop-patience 5 \
  --eval-cuda-device 0 \
  --regenerate-baseline True \
  --closed-loop-eval False \
  --closed-loop-routes leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --closed-loop-team-agent Bench2DriveZoo/team_code/vad_b2d_agent.py \
  --closed-loop-port 30000 \
  --closed-loop-tm-port 50000 \
  --closed-loop-gpu-rank 0 \
  --closed-loop-planner-type only_traj \
  --closed-loop-is-bench2drive True \
  --time-horizon 3 \
  --occ-output-dir baseline/VAD/vad_occ_cache \
  --collision-num-workers 13 \
  > logs/run_openloop_repair_6.out 2>&1 &

wait
