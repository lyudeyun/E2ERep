#!/usr/bin/env bash
set -euo pipefail

REPO="rethinklab/Bench2Drive-Map"
OUTDIR="${1:-./Bench2Drive-Map}"

# 1) 安装 hf CLI（来自 huggingface_hub）
python3 -m pip install -U "huggingface_hub[cli]"  >/dev/null

# 2) （可选）如果是私有仓库/限流，先登录
# hf auth login

mkdir -p "$OUTDIR"

# 3) 下载：默认全量
#    若只想下载部分文件，取消下面“全量”这一行，改用 include 示例
hf download --repo-type dataset "$REPO" --local-dir "$OUTDIR"

# ---- 只下载部分文件（示例）----
# hf download --repo-type dataset "$REPO" --local-dir "$OUTDIR" \
#   --include "Town11_HD_map.npz" --include "Town12_HD_map.npz"
