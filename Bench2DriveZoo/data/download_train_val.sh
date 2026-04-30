#!/bin/bash
set -euo pipefail

# 1) Try activating conda env b2d_zoo
if command -v conda &>/dev/null; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate b2d_zoo
fi

# 2) Ensure hf CLI is available (huggingface_hub)
if ! command -v hf &>/dev/null; then
  echo "hf CLI not found. Trying to install huggingface_hub..."
  python -m pip install --user huggingface_hub
fi

echo "Using hf from: $(command -v hf)"

# 3) Output directory and mode
# Usage:
#   ./download_train_val.sh       # 50 validation clips only
#   ./download_train_val.sh all   # Full dataset (large download)
MODE="${1:-val}"
if [[ "$MODE" == "all" ]]; then
  OUT_DIR="Bench2Drive-all"
else
  OUT_DIR="Bench2Drive-val"
fi
mkdir -p "$OUT_DIR"

if [[ "$MODE" == "all" ]]; then
  echo "Downloading full dataset to: $OUT_DIR ..."
  hf download rethinklab/Bench2Drive \
    --repo-type dataset \
    --local-dir "$OUT_DIR" \
    --force-download
  echo "✅ Full dataset downloaded to: $OUT_DIR"
  exit 0
fi

# 4) These 50 validation clips (no v1/ prefix)
clips=(
"ParkingCrossingPedestrian_Town13_Route545_Weather25"
"OppositeVehicleTakingPriority_Town04_Route214_Weather6"
"DynamicObjectCrossing_Town02_Route11_Weather11"
"AccidentTwoWays_Town12_Route1115_Weather23"
"VehicleTurningRoute_Town15_Route504_Weather10"
"ParkingExit_Town12_Route922_Weather12"
"SignalizedJunctionLeftTurn_Town04_Route173_Weather26"
"EnterActorFlow_Town03_Route132_Weather2"
"HighwayExit_Town06_Route312_Weather0"
"VanillaSignalizedTurnEncounterRedLight_Town15_Route491_Weather23"
"CrossingBicycleFlow_Town12_Route977_Weather15"
"OppositeVehicleRunningRedLight_Town04_Route180_Weather23"
"VanillaSignalizedTurnEncounterRedLight_Town07_Route359_Weather21"
"ParkingCutIn_Town13_Route1343_Weather1"
"ParkedObstacle_Town06_Route282_Weather22"
"TJunction_Town06_Route306_Weather20"
"PedestrianCrossing_Town13_Route747_Weather19"
"VehicleTurningRoutePedestrian_Town15_Route445_Weather11"
"ConstructionObstacle_Town12_Route78_Weather0"
"HazardAtSideLaneTwoWays_Town12_Route1151_Weather7"
"ControlLoss_Town04_Route170_Weather14"
"MergerIntoSlowTrafficV2_Town12_Route857_Weather25"
"DynamicObjectCrossing_Town01_Route3_Weather3"
"SignalizedJunctionRightTurn_Town03_Route118_Weather14"
"BlockedIntersection_Town03_Route135_Weather5"
"MergerIntoSlowTraffic_Town06_Route317_Weather5"
"NonSignalizedJunctionRightTurn_Town03_Route126_Weather18"
"ParkedObstacleTwoWays_Town13_Route1333_Weather26"
"ConstructionObstacleTwoWays_Town12_Route1093_Weather1"
"TJunction_Town05_Route260_Weather0"
"NonSignalizedJunctionLeftTurn_Town07_Route342_Weather3"
"HighwayCutIn_Town12_Route1029_Weather15"
"HazardAtSideLane_Town10HD_Route373_Weather9"
"YieldToEmergencyVehicle_Town04_Route166_Weather10"
"HardBreakRoute_Town01_Route32_Weather6"
"SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2"
"ConstructionObstacle_Town10HD_Route74_Weather22"
"ControlLoss_Town10HD_Route378_Weather14"
"Accident_Town05_Route218_Weather10"
"InterurbanActorFlow_Town12_Route1291_Weather1"
"LaneChange_Town06_Route307_Weather21"
"InvadingTurn_Town02_Route95_Weather9"
"VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9"
"StaticCutIn_Town05_Route226_Weather18"
"VehicleOpensDoorTwoWays_Town12_Route1203_Weather7"
"VehicleTurningRoutePedestrian_Town15_Route481_Weather19"
"VanillaSignalizedTurnEncounterGreenLight_Town07_Route354_Weather8"
"NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13"
"InterurbanAdvancedActorFlow_Town06_Route324_Weather2"
"ParkedObstacle_Town10HD_Route372_Weather8"
)

# 5) Download loop
for clip in "${clips[@]}"; do
  fname="${clip}.tar.gz"
  echo "Downloading $fname ..."
  hf download rethinklab/Bench2Drive \
    --repo-type dataset \
    --include "$fname" \
    --local-dir "$OUT_DIR" \
    --force-download
  echo "Done: $fname"
done

echo "✅ All clips downloaded to: $OUT_DIR"
