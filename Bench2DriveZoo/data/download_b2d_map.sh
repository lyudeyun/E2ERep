#!/usr/bin/env bash
set -euo pipefail

REPO="rethinklab/Bench2Drive-Map"
OUTDIR="${1:-./Bench2Drive-Map}"

# 1) Install hf CLI (from huggingface_hub)
python3 -m pip install -U "huggingface_hub[cli]"  >/dev/null

# 2) (Optional) Log in for private repos or rate limits
# hf auth login

mkdir -p "$OUTDIR"

# 3) Download: full dataset by default
#    For a subset, comment the line below and use the --include example instead
hf download --repo-type dataset "$REPO" --local-dir "$OUTDIR"

# ---- Partial download example ----
# hf download --repo-type dataset "$REPO" --local-dir "$OUTDIR" \
#   --include "Town11_HD_map.npz" --include "Town12_HD_map.npz"
