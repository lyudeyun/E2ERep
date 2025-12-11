#!/bin/bash
set -e  # 一旦有命令失败就退出

PYTHON_BIN=python3

# 1) 确认 python3 存在
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found in PATH."
  exit 1
fi

# 2) 检查 / 安装 huggingface_hub
if ! "$PYTHON_BIN" -m pip show huggingface_hub > /dev/null 2>&1; then
  echo "huggingface_hub is not installed. Installing now..."
  "$PYTHON_BIN" -m pip install --user huggingface_hub
else
  echo "huggingface_hub is already installed."
fi

# 3) 确保 hf 命令可用（新版 CLI）
if ! command -v hf >/dev/null 2>&1; then
  # 你现在的 hf 在这个目录里：~/Library/Python/3.9/bin
  HF_BIN="$HOME/Library/Python/3.9/bin/hf"
  if [ -x "$HF_BIN" ]; then
    echo "Adding $HOME/Library/Python/3.9/bin to PATH for this script..."
    export PATH="$HOME/Library/Python/3.9/bin:$PATH"
  else
    echo "Error: 'hf' CLI not found."
    echo "Try:  python3 -m pip install --user huggingface_hub"
    exit 1
  fi
fi

# 4) 创建存放数据的目录
mkdir -p Bench2Drive-mini

# 5) 需要下载的文件列表
FILES=(
  "HardBreakRoute_Town01_Route30_Weather3.tar.gz"
  "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz"
  "Accident_Town03_Route156_Weather0.tar.gz"
  "YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz"
  "ConstructionObstacle_Town05_Route68_Weather8.tar.gz"
  "ParkedObstacle_Town10HD_Route371_Weather7.tar.gz"
  "ControlLoss_Town11_Route401_Weather11.tar.gz"
  "AccidentTwoWays_Town12_Route1444_Weather0.tar.gz"
  "OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz"
  "VehicleTurningRoute_Town15_Route443_Weather1.tar.gz"
)

# 6) 逐个下载
for f in "${FILES[@]}"; do
  echo "Downloading $f ..."
  hf download rethinklab/Bench2Drive \
    --repo-type dataset \
    "$f" \
    --local-dir Bench2Drive-mini \
    --force-download
done

echo "✅ All downloads finished."