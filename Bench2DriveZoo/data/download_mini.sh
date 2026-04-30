#!/bin/bash
set -e  # Exit on first failure

PYTHON_BIN=python3

# 1) Ensure python3 exists
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found in PATH."
  exit 1
fi

# 2) Check / install huggingface_hub
if ! "$PYTHON_BIN" -m pip show huggingface_hub > /dev/null 2>&1; then
  echo "huggingface_hub is not installed. Installing now..."
  "$PYTHON_BIN" -m pip install --user huggingface_hub
else
  echo "huggingface_hub is already installed."
fi

# 3) Ensure hf CLI is available
if ! command -v hf >/dev/null 2>&1; then
  # Typical user install path on macOS
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

# 4) Create output directory
mkdir -p Bench2Drive-mini

# 5) Files to download
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

# 6) Download each file
for f in "${FILES[@]}"; do
  echo "Downloading $f ..."
  hf download rethinklab/Bench2Drive \
    --repo-type dataset \
    "$f" \
    --local-dir Bench2Drive-mini \
    --force-download
done

echo "✅ All downloads finished."