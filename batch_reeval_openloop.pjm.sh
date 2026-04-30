#!/bin/bash
# Batch re-run open-loop evaluation — PJM job script
# Usage: edit EXP_DIR / EVAL_DATASET below, then: pjsub batch_reeval_openloop.pjm.sh

#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L "elapse=24:00:00"
#PJM -j
#PJM -o batch_reeval_openloop_job_%j.log
#PJM -m e
# Replace with your address for job-end mail from the scheduler:
#PJM --mail-list recipient@example.com

set -euo pipefail

# ==== Tunables (edit as needed) ====
# Experiment root (each subdir is one experiment; rescan for incomplete open-loop eval)
EXP_DIR="${EXP_DIR:-/home/pj25001076/ku50002427/git/B2DRepair/uniad_base_Arachne_v2_DE_results}"
# Eval dataset PKL (relative to B2DRepair repo root)
EVAL_DATASET="${EVAL_DATASET:-Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl}"
# GPU for evaluation
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
# Parallel eval workers on this GPU (default 1 = serial)
JOBS="${JOBS:-6}"
# Occupancy cache dir (optional; omit env to skip passing --occ-output-dir)
OCC_OUTPUT_DIR="${OCC_OUTPUT_DIR:-baseline/UniAD/uniad_occ_cache}"
# Dry-run only when DRY_RUN=1
DRY_RUN="${DRY_RUN:-0}"

# ==== Environment (enable module load if needed) ====
# module load cuda

# ==== Conda ====
source /home/pj25001076/ku50002427/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ==== Repo root ====
cd /home/pj25001076/ku50002427/git/B2DRepair || exit 1

# ==== Logs ====
mkdir -p logs

# ==== Extra CLI args ====
EXTRA_ARGS=()
if [[ -n "${OCC_OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--occ-output-dir "$OCC_OUTPUT_DIR")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

# ==== Run batch re-eval ====
echo "=========================================="
echo "批量重跑开环评估"
echo "  EXP_DIR=$EXP_DIR"
echo "  EVAL_DATASET=$EVAL_DATASET"
echo "  EVAL_CUDA_DEVICE=$EVAL_CUDA_DEVICE"
echo "  JOBS=$JOBS"
echo "=========================================="
echo "即将执行的命令:"
printf 'CUDA_VISIBLE_DEVICES=%s python3 batch_reeval_openloop.py --exp-dir %q --eval-dataset %q --eval-cuda-device %q --jobs %q' \
  "$EVAL_CUDA_DEVICE" "$EXP_DIR" "$EVAL_DATASET" "$EVAL_CUDA_DEVICE" "$JOBS"
printf ' %s' "${EXTRA_ARGS[@]}"
echo
echo "=========================================="

CUDA_VISIBLE_DEVICES=${EVAL_CUDA_DEVICE} \
python3 batch_reeval_openloop.py \
  --exp-dir "$EXP_DIR" \
  --eval-dataset "$EVAL_DATASET" \
  --eval-cuda-device "$EVAL_CUDA_DEVICE" \
  --jobs "$JOBS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/batch_reeval_openloop_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "批量重评结束"
echo "=========================================="
