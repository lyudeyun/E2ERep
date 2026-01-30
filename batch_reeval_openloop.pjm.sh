#!/bin/bash
# 批量重新运行开环评估 - PJM 超算作业脚本
# 用法: 修改下方 EXP_DIR / EVAL_DATASET 等变量后，提交: pjsub batch_reeval_openloop.pjm.sh

#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L "elapse=24:00:00"
#PJM -j
#PJM -o batch_reeval_openloop_job_%j.log
#PJM -m e
#PJM --mail-list lyudeyun@gmail.com

set -euo pipefail

# ==== 可调参数（按需修改）====
# 实验根目录（其下每个子目录为一个实验，会扫描并重跑未完成开环评估的）
EXP_DIR="${EXP_DIR:-/home/pj25001076/ku50002427/data/uniad_base_Arachne_v2_DE_results}"
# 评估数据集 PKL（相对路径相对于 B2DRepair 仓库根）
EVAL_DATASET="${EVAL_DATASET:-Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl}"
# 使用的 GPU
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
# 并行评估数（单卡下同时跑几个，默认 1 串行）
JOBS="${JOBS:-6}"
# occupancy 缓存目录（可选，不设则不传该参数）
OCC_OUTPUT_DIR="${OCC_OUTPUT_DIR:-baseline/UniAD/uniad_occ_cache}"
# 仅打印不执行：设为 1 则 dry-run
DRY_RUN="${DRY_RUN:-0}"

# ==== 环境（按需启用 module）====
# module load cuda

# ==== Conda 环境 ====
source /home/pj25001076/ku50002427/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ==== 进入工程 ====
cd /home/pj25001076/ku50002427/git/B2DRepair || exit 1

# ==== 日志目录 ====
mkdir -p logs

# ==== 构建参数 ====
EXTRA_ARGS=()
if [[ -n "${OCC_OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--occ-output-dir "$OCC_OUTPUT_DIR")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

# ==== 运行批量重评 ====
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
